# EdgeVision

**Tek web kamerasıyla metre cinsinden mesafe, nesne tanıma ve engel kaçınma.** Derinlik sensörü, stereo kamera, LiDAR ya da ToF yok — yalnızca sıradan bir USB kamera.

Sistem iki soruyu birlikte cevaplar: *ne kadar uzakta* (metrik derinlik ağı) ve *ne* (YOLO26 segmentasyon). Ekranda en yakın nesnenin adı ve mesafesi yazar: `EN YAKIN NESNE: insan 0.53 m`

## Hızlı başlangıç

```bash
pip install -r requirements.txt
```

```bash
pip install ultralytics
```

```bash
python setup_models.py --depth yolo26
```

```bash
python main.py
```

Model dosyaları depoya girmez (~130 MB); `setup_models.py` onları üretip yerine koyar. Ayrıntılar: [Kurulum](#8-kurulum). Eksik model varsa sistem çökmez, ne yapılacağını yazar.

Pencerede: `q` çıkış · `c` kalibrasyon · `d` sağ panel · `s` kare kaydet · tıklama = o pikselin mesafesi

## Ölçülen sonuçlar

| | |
|---|---|
| Ana döngü hızı | **30–31 FPS** (kamera-bağımlı) |
| İşlem + çizim maliyeti | 4.88 ms/kare |
| Derinlik — Depth Anything V2 (518) | 37–42 ms, 18–21 FPS (DirectML) |
| Derinlik — YOLO26 (480×640) | **5–6.6 ms**, 30 FPS (DirectML) |
| Tanıma — YOLO26-seg (640) | 40–53 ms, 20–27 FPS (CPU, 12 iş parçacığı) |
| Tanıma katmanının ana döngüye maliyeti | **Yok** (20.2 → 20.2 FPS) |
| Etiket kararlılığı (izleme sonrası) | 58 → **6** değişim / ~430 kare |
| Sabit cisim mesafe gürültüsü | ±1.9 → **±0.5 cm** |
| Küresel ölçek kayması (çapa öncesi/sonrası) | −50.2% → **+15.1%** |
| El tanıma (MediaPipe) | 11.2 ms, ana döngüye maliyeti **yok** |

---

## 1. Sistem Nedir, Ne Yapar

**Amaç:** Sıradan bir web kamerasıyla, ek hiçbir donanım kullanmadan, görüş alanındaki her yön için **metre cinsinden mesafe** üretmek ve bunu engel kaçınma komutlarına çevirmek.

**Donanım kısıtı:** Kesinlikle tek kamera. Sistem başka hiçbir sensöre bağlı değildir.

**Temel yaklaşım:** Mesafe tek bir kaynaktan gelir — metrik derinlik sinir ağı, her piksel için doğrudan metre değeri üretir. Klasik monoküler geometri yığını (zemin düzlemi varsayımları, nesne boyutu öğrenme, ArUco kalibrasyonu, MiDaS füzyonu) tamamen kaldırılmıştır çünkü bu model onları gereksiz kılar. İki motor seçilebilir: Depth Anything V2 (daha geniş dinamik aralık) veya YOLO26 derinlik (8 kat hızlı, kalibrasyonsuz).

**Tanıma katmanı:** Derinlik ağı "ne kadar uzakta" sorusunu cevaplar ama cismin **ne** olduğunu bilmez. YOLO26 segmentasyon (`yolo26n-seg.onnx`, COCO 80 sınıf) bu boşluğu kapatır ve cismin **maskesini** verir; mesafe o maskedeki derinlikten hesaplanır. Açık sözlüklü seçenek (`yoloe-26n-seg.onnx`) COCO'da olmayan sınıfları da (kapı, duvar, merdiven) metin komutuyla tanır.

**El katmanı:** COCO'da "el" sınıfı yoktur, model eli "insan" sayar. MediaPipe HandLandmarker eli ayrı tanır ve aynı bölgedeki "insan" kutusunun yerine geçer. Açık sözlüklü model de denendi ve tutmadı: 503 karede `hand` için en yüksek güven 0.00 kaldı. Katman opsiyoneldir — `mediapipe` kurulu değilse sessizce kapanır.

**Ölçek çapası:** Monoküler derinlik modelleri sahne değişince **tüm haritayı** yeniden ölçekler. Ölçüldü: kareye yakın bir cisim girdiğinde görüntünün hiç değişmeyen yarısındaki mesafe %50 kaydı. Sahnenin uzak kısmı referans alınarak bu kayma %15'e indirilir.

---

## 2. Mimari ve Veri Akışı

```
 Kamera (ayrı iş parçacığı, son kare)
   │
   ▼
 VisionPipeline.process()
   │
   ├─► küçült (640x480 → 320x240)
   │
   ├─► depth.submit(kare) ──────► MetricDepthNet (AYRI İŞ PARÇACIĞI)
   │                                  │  ONNX çıkarımı (~30 ms)
   │                                  │  kalibrasyon (afin veya tablo)
   │                                  │  median denoise
   │                                  ▼
   ├─◄── depth.get_depth() ◄──── yayınlanan harita + sürüm sayacı
   │
   ├─► yolo.submit(kare) ───────► YoloDetector (AYRI İŞ PARÇACIĞI)
   │                                  │  YOLO26-seg ONNX (~50 ms, CPU)
   │                                  │  NMS → kutular + COCO etiketleri
   │                                  ▼
   ├─◄── yolo.get_detections() ◄─ son yayınlanan kutular
   │
   ├─► GroundRemover: zemin düzlemi uydur, zemin maskesi üret
   │
   ├─► fuse_distance(): her kutunun mesafesi = kutu çekirdeğindeki derinlik
   │
   ├─► ObstacleScanner.scan(): sütun başına en yakın yüzey (zemin hariç)
   │
   ├─► ScanSmoother: hızlı-yaklaşma / yavaş-uzaklaşma histerezisi
   │
   ├─► ObstacleScanner.cluster(): komşu sütunları engellere kümele
   │
   ├─► ön-plan/arka-plan ayrımı (en yakın N küme)
   │
   ├─► ObstacleReport  ──► AvoidancePlanner.plan() ──► AvoidanceCommand
   │
   └─► viz.compose_view(): kamera + nesne kutuları | derinlik haritası
```

### İki iş parçacığı neden şart

Derinlik ağı ViT tabanlıdır ve tek başına ~25-30 FPS koşar. Eğer ana döngüyü bloke etseydi tüm sistem o hıza düşerdi. Ağ ayrı bir iş parçacığında sürekli koşar ve en son haritayı paylaşımlı olarak yayınlar; ana döngü bu haritayı bloke olmadan okur. Böylece ana döngü kamera hızında (30 FPS) akarken derinlik arka planda tazelenir.

Tanıma ağı aynı deseni izler (üçüncü iş parçacığı). Ölçüldü: tanıma eklendikten sonra ana döngü FPS'i **değişmedi** (20.2 → 20.2) — çıkarım maliyeti ana döngüye hiç yansımaz. Mesafeler ise her karede o anki haritadan yeniden hesaplanır: harita, kutulardan daha sık tazelenir.

**Sağlayıcı çakışması:** İki DirectML oturumu aynı anda ayakta kalamıyor. Derinlik ağı GPU'yu (DirectML) tutar, YOLO **CPU'da** koşar — 27 ms zaten fazlasıyla yeter.

---

## 3. Dosya Dosya Referans

### `main.py` — Giriş noktası

Argüman ayrıştırma, kamera döngüsü, pencere yönetimi, klavye/fare olayları.

| Çalışma modu | Komut |
|---|---|
| Pencere (varsayılan) | `python main.py` |
| Derinlik motoru karşılaştırma | `python compare_depth.py` |
| Video dosyası | `python main.py --source test.mp4` |
| Robot köprüsü (JSON) | `python main.py --json` |
| Penceresiz, konsol durumu | `python main.py --headless` |

**Klavye:** `q` çıkış · `s` kare kaydet · `d` sağ panel değiştir · `c` kalibrasyon modu
**Fare:** Tıklama → o pikselin mesafesi (kalibre + ham değer birlikte)

**Ekranda ne görünür:** Tanınan her nesne kutulanır ve adlandırılır; en yakın olan sarıyla vurgulanır ve üstteki bantta `EN YAKIN NESNE: <ad> <metre>` olarak yazılır. `--headless` modunda aynı bilgi konsola düşer (`nesne=insan 0.53m`), `--json` modunda ise `detections` ve `nearest_named` alanlarıyla robot köprüsüne aktarılır.

**Önemli detay:** Kenar haritası yalnızca sağ panelde gösterilecekse hesaplanır (`want_edges`). Gösterilmeyecek bir ürünü hesaplamak boşa CPU yakar.

---

### `edge_vision/config.py` — Yapılandırma

YAML dosyasını dataclass'lara yükler. Bilinmeyen anahtarlar sessizce atılır (eski config dosyaları kırılmaz).

| Sınıf | Görev |

| `CameraConfig` | Kaynak, çözünürlük, FPS, yatay görüş açısı |
| `ProcessingConfig` | İşleme genişliği, ROI üst sınırı, sütun adımı |
| `EdgeConfig` | Canny kenar tespiti (yalnız görselleştirme) |
| `DepthConfig` | Model yolu, girdi boyutu, sağlayıcılar, **kalibrasyon**, renk skalası |
| `YoloConfig` | Tanıma modeli yolu, güven/NMS eşikleri, sağlayıcılar |
| `AvoidanceConfig` | Mesafe eşikleri, merkez koridor, izlenen engel sayısı |
| `SmoothingConfig` | Zamansal histerezis |
| `GroundConfig` | Zemin düzlemi ayıklama parametreleri |
| `AppConfig` | Hepsini toplayan kök |

`load_config(path)` → `AppConfig`

---

### `edge_vision/camera.py` — Kamera katmanı

**`ThreadedCamera`** — Canlı kamerada arka planda sürekli okur ve **yalnızca en son kareyi** saklar. Eski kareler kuyrukta birikmez.

---

### `edge_vision/intrinsics.py` — Kamera iç parametreleri

**`Intrinsics`** (fx, fy, cx, cy) ve **`intrinsics_from_hfov()`**.

Metrik derinlik ağı Z'yi (ileri mesafe) doğrudan verdiği için geometriden tek ihtiyaç yatay açı ve yanal konumdur:

```
X = Z · (u − cx) / fx        yön = atan(u − cx, fx)
```

---

### `edge_vision/metric_depth.py` — Derinlik ağı 
**`MetricDepthNet`** — Depth Anything V2 ONNX modelini ayrı bir iş parçacığında koşturur.

| Metot | Görev |
|---|---|
| `start(out_hw)` | Arka plan iş parçacığını başlatır |
| `submit(bgr)` | Ana döngü her karede son kareyi bırakır |
| `_infer(bgr)` | Ön işleme → ONNX → kalibrasyon → median |
| `get_depth()` | Kalibre haritayı döndürür (kopya) |
| `get_raw_depth()` | **Ham** haritayı döndürür (kalibrasyon için) |
| `set_calibration_table(points)` | Parçalı kalibrasyon tablosunu kurar |
| `version` | Yeni harita sayacı (gereksiz iş atlamak için) |
| `visualize(depth_m)` | Renkli derinlik paneli üretir |

**Sağlayıcı seçimi:** DirectML (herhangi bir DirectX 12 GPU, dahili Intel/AMD dahil) varsa öncelikli; yoksa CPU. Windows'ta DirectML CPU'ya göre kat kat hızlıdır.

**Kritik sıra kısıtı:** `set_calibration_table()` `_lock` kullanır, bu yüzden `__init__` içinde kilitler kurulduktan **sonra** çağrılmalıdır. Bu sırayla bir kez hata yapıldı ve düzeltildi.

---

### `edge_vision/object_detector.py` — Nesne tanıma

**`Detection`** (dataclass) — Tek bir tanıma: sınıf, Türkçe etiket, güven, kutu, mesafe.

**`YoloDetector`** — tanıma modelini ayrı bir iş parçacığında koşturur (derinlik ağıyla aynı desen: `start` / `submit` / `get_detections`).

| Metot | Görev |
|---|---|
| `start()` | Arka plan iş parçacığını başlatır |
| `submit(bgr)` | Ana döngü her karede son kareyi bırakır |
| `_preprocess(frame)` | Letterbox (oran korunur, gri dolgu) → NCHW float |
| `_infer(frame)` | ONNX → sınıf skorları → NMS → kare koordinatına geri dönüşüm |
| `get_detections()` | Son yayınlanan kutu listesi |

**Neden letterbox:** Kareyi doğrudan 640×640'a esnetmek en-boy oranını bozar; YOLO eğitimi letterbox varsayar, esnetilmiş girdide kutular kayar.

**Girdi boyutu oturumdan okunur:** modeller sabit boyutlu dışa aktarılır. Config'te başka bir değer yazsa bile oturumun bildirdiği boyut kullanılır — aksi halde ONNX anlaşılmaz bir şekil hatası verirdi.

### `edge_vision/object_detector.py` → `DetectionTracker` — etiket kararlılığı

**Çözdüğü problem:** YOLO her kareyi **bağımsız** işler. İki sınıfın skoru birbirine yakınsa (bardak / insan) etiket kare kare zıplar; ekranda saniyede birkaç kez değişir. Ayrıca tek karelik hayalet kutular (ölçüldü: boş sahnede "tuvalet") belirip kaybolur.

**Çözüm:** Kutular kareler arasında IoU ile eşlenir, her izin son N karesindeki sınıf oyları tutulur ve **gösterilen etiket oylamayla** belirlenir.

| Kural | Neden |
|---|---|
| `vote_frames` (15) çoğunluk oyu | Tek karelik yanlış tahmin çoğunluğu deviremez |
| `switch_ratio` (1.5x) **üstünlük şartı** | Düz çoğunluk yetmiyor: %70/%30 dağılımda bile pencere kayarken çoğunluk el değiştiriyor (ölçüldü: 80 karede 3 kez). Üstünlük şartıyla 1'e indi |
| `min_hits` (3) | Hayalet kutu ekrana hiç gelmez |
| `max_miss` (5) | Kısa kayboluşta kutu tutulur; yoksa cisim yanıp sönüyor |
| Ölü bantlı EMA | Cisim dururken okuma sabit, hareket edince hızlı takip |

**Neden sürüm sayacı:** İzleyici yalnızca **yeni bir çıkarım** yayınlandığında beslenir. Ana döngü ağdan hızlı döndüğü için `get_detections()` aynı listeyi tekrar tekrar verir; her karede oylamak oyları yapay olarak şişirirdi.

**Ölçülen sonuçlar:**

| | Ham YOLO | İzlenen |
|---|---|---|
| Sentetik %70/%30 sınıf zıplaması | 32 değişim / 80 kare | **1** / 78 |
| Canlı sahne, etiket kümesi değişimi | 58 / 426 çıkarım | **6** / 429 kare |
| Hayalet tespit ("tuvalet") | görünüyor | **hiç ekrana gelmedi** |
| Sabit cisim mesafe gürültüsü | ±1.9 cm | **±0.5 cm** |
| Ani yaklaşma (0.80 → 0.40 m) tepkisi | — | 3 kare (0.15 s) |

Ölü bant güvenliği bozmaz: **yaklaşma** hareket sayılır ve hızlı alfa ile anında geçer; bastırılan yalnızca duran cismin kare-kare oynamasıdır.

---

**İki model biçimi** desteklenir, hangisi yüklendiği oturumdan anlaşılır (config'te bayrak yok — yanlış ayarlanmış bir bayrak sessizce saçma kutular üretirdi):

| Model | Çıktı | Çözümleme |
|---|---|---|
| `yolo11n.onnx` | `(1, 84, 8400)` | sınıf skorları + **NMS gerekir** |
| `yolo26n-seg.onnx` | `(1, 300, 38)` + `(1, 32, 160, 160)` | uçtan uca, **NMS yok**, + maske |

YOLO26 satır düzeni: `[x0, y0, x1, y1, güven, sınıf, 32 maske katsayısı]`. Çözümleme ultralytics'in kendi çıktısıyla doğrulandı — aynı kutu (`[1,83,259,237]` ↔ `[2,82,258,240]`), aynı sınıf, maske alanı %2 içinde.

**Maske üretimi:** katsayılar × prototipler → sigmoid → kutuya kırp. Prototip ızgarası (160×160) letterbox tuvalinin ölçekli halidir; tüm maskeyi 640×640'a büyütüp kırpmak yerine **önce prototip uzayında kırpılır** — aynı sonuç, dörtte bir maliyet.

---

**`fuse_distance(det, depth_m, ground_mask)`** — Nesneyi metrik mesafeye çevirir:

| Karar | Gerekçe |
|---|---|
| **Maske varsa doğrudan o kullanılır** | Kestirim değil, cismin gerçek pikselleri. Ölçüldü: bir insanda kutunun yalnızca **%54'ü** cisim, gerisi arka plan |
| **Maske yolunda zemin filtresi UYGULANMAZ** | Maske zaten "bu pikseller bu cisme ait" diyor; üstüne sezgisel düzlem filtresi koymak yalnızca gerçek pikselleri siler. Ölçüldü: zemin kilitliyken bir insanın maskesinin **%63'ü** zemin sayılıp atılıyordu |
| Maske yoksa kutunun **orta %50'si** örneklenir | Kutu köşeleri neredeyse her zaman arka planı içerir; cismin gövdesi ortadadır |
| Zemin maskesi dışlanır | Sandalye/masa kutusunun altındaki yer mesafeyi yakına çekiyordu |
| Medyan | Tek gürültülü piksel sonucu kaydıramaz |
| Çekirdek boşsa tüm kutuya düş | Küçük/tamamı zemin kutularda sonuç yine de üretilsin |

---

### `edge_vision/hand_detector.py` — El tanıma

COCO'da **"el" sınıfı yoktur**; model eli "insan" sayar. MediaPipe HandLandmarker eli ayrı tanır ve 21 eklem noktasından kutu üretir.

**Önce açık sözlüklü model denendi ve tutmadı:** YOLOE'nin sözlüğüne `hand` eklenip yeniden dışa aktarıldı, **503 karede en yüksek güven 0.00** çıktı — model o sınıf için hiç öneri üretmedi.

**Birleştirme kuralı** (`VisionPipeline._merge`): el tespiti, kendisiyle `IoU > 0.5` örtüşen **"insan" kutusunun yerine geçer**. Kameraya uzatılan elde YOLO'nun "insan" dediği kutu zaten elin kendisidir; ikisi birden gösterilseydi aynı cisim iki kez sayılır ve en yakın nesne "insan" diye okunurdu.

**Opsiyoneldir:** `HandDetector.create()` paket ya da `.task` dosyası yoksa `None` döner, katman sessizce kapanır. Bu yüzden `mediapipe` zorunlu bağımlılık değildir — clone eden birinin kurulumu bunun yüzünden bozulmaz.

Ölçüldü: 11.2 ms, ana döngü FPS'ine etkisi **yok** (kendi iş parçacığında).

---

### `edge_vision/pipeline.py` → `ScaleAnchor` — küresel ölçek kayması

**Çözdüğü problem:** Monoküler derinlik modeli ölçeği tüm sahneden çıkarır. Kadraja yakın bir cisim girdiğinde **haritanın tamamı** yeniden ölçeklenir — kullanıcının gördüğü belirti: *"sandalyeyi gösteriyordum, elimi uzatınca sandalyenin de rengi değişti"*.

Kontrollü ölçüm (kareye yapay bir yakın cisim eklenip görüntünün **hiç dokunulmamış** yarısına bakıldı):

| Motor | Değişmeyen yarı, cisimsiz → cisimli | Kayma |
|---|---|---|
| `dav2` | 1.666 → 0.835 (ham) | **−49.9%** |
| `yolo26` | 2.865 → 1.299 (ham) | **−54.6%** |

**Her iki motorda da var** — motor değiştirmek çözmez, monoküler derinliğin doğasında.

**Çözüm:** Sahnenin uzak kısmı (`ref_pct` = 85. yüzdelik) gerçekte sabit kaldığı için kaymanın göstergesidir. Referans ilk karelerde kilitlenir, sonra `k = referans / ölçüm` oranı EMA ile yumuşatılarak haritaya uygulanır.

| | Değişmeyen bölgedeki hata |
|---|---|
| Çapa kapalı | −50.2% |
| Çapa açık | **+15.1%** |

Kalan +%15, kaymanın saf çarpımsal olmamasından: model uzak menzili orta menzilden daha çok sıkıştırıyor, tek katsayı tamamını gideremiyor. Yine de hata **3.3 kat** azalıyor.

Cismin **gerçek** yaklaşması arka planı değiştirmediği için bastırılmaz; düzeltilen yalnızca modelin kaymasıdır.

---

### `edge_vision/stability.py` — Ani değişimler (ışık, bulanıklık, sahne kesmesi)

Derinlik ağı ve YOLO her kareyi bağımsız yorumlar; görüntü bir anda değişince
çıktılar da bir anda kayar. `SceneMonitor` her karede 80x60 gri görüntüden
(<0.5 ms) şu olayları çıkarır:

| Olay | Ölçüt | Hattın tepkisi |
|------|-------|----------------|
| `isik` | kareden kareye bağıl parlaklık > `light_jump` ya da z > `z_max` | ölçek çapası güncellenmez |
| `bulanik` | keskinlik < `blur_ratio` × öğrenilen ortalama | ölçek çapası güncellenmez |
| `sarsinti` | parlaklıktan bağımsız kare farkı z > `z_max` | iz mesafesi kapısı 2 kat sabırlı |
| `degisim` | yapısal benzerlik (NCC) < `cut_corr` | aday sahne sayılır |
| `kesme` | yeni sahne `cut_persist` kare tutarlı | ölçek çapası referansı yeni sahneye göre yeniden kurulur |

Eşikler sabit değildir: her öznitelik için çevrimiçi EWMA ortalama/varyans
öğrenilir (denetimsiz, çevrimiçi anomali tespiti), z-skoru o ortamın kendi
gürültüsüne göre hesaplanır. Olaydan sonra `settle_frames` kare boyunca sahne
"kararsız" sayılır. Kameranın önünden geçen el gibi kısa örtmeler kesme sayılmaz.

**Sıçrama kapısı** (`DetectionTracker.set_distance`): bir izin mesafesi tek
karede `jump_ratio`'dan (%35) fazla değişirse hemen kabul edilmez; yeni değer
`jump_persist` kare tutarlı kalırsa gerçek hareket sayılıp doğrudan atanır.
Kaçınma komutunu üreten sütun taraması bu kapıdan GEÇMEZ — yaklaşan engel
güvenlik için gecikmesiz görülür.

### `edge_vision/log.py` — Loglama ve telemetri

Tüm modüller `edge_vision.*` logger'larına yazar: konsol (stderr) + dönen
dosya `logs/edgevision.log`. `--json` kipinde stdout temiz kalır. Arka plan iş
parçacıkları (derinlik/YOLO/el) hata verince artık ölmez: hata loglanır,
döngü devam eder.

```powershell
python main.py --headless --log-level DEBUG --telemetry
```

`--telemetry` kare başına bir JSONL satırı yazar (`logs/telemetry_*.jsonl`):
sahne öznitelikleri, olaylar, ham derinlik medyanı ve sıçraması, çapa
katsayısı, en yakın engel/nesne. Ani değişim sorunlarını sonradan incelemek
ve gerekirse denetimli bir model (ör. ölçek düzeltmesini öğrenen bir
regresör) eğitmek için veri setidir.

### `edge_vision/ground.py` — Zemin düzlemi ayıklama

**`GroundRemover`**

**Çözdüğü problem:** Sütun başına "en yakın yüzey" taraması ZEMİNİ yakalar. Kameranın önündeki yer, çoğu sütunda gerçek cisimlerden daha yakındır. Ölçüldü: sütunların **%78'i** en yakın pikselini görüntünün en alt %10'unda buluyordu. Sonuç: düz bir odada sistem her yerde sahte ~0.7 m'lik engel görüyor, tüm sütunlar tek kümede birleşiyor (ekranda tam genişlikte tek çizgi) ve sandalye/insan gibi gerçek cisimler zeminin içinde kayboluyordu.

**Çözüm:** Metrik derinlik + kamera iç parametreleriyle her pikselin 3B konumu (X, Y, Z) hesaplanır; zemin bir düzlem olarak sağlam biçimde uydurulur; düzleme yakın veya altında kalan pikseller taramadan çıkarılır. Geriye yalnızca yerden **yükselen** şeyler kalır.

| Özellik | Detay |
|---|---|
| Düzlem uydurma | `a·X + b·Y + c·Z = 1`, en küçük kareler + 4 tur kırpılmış aykırı-nokta ayıklama |
| Eşik | **Yüzdelik** (her turda %60 hayatta kalıyor), sabit metre değil |
| Mesafeye uyarlanan marj | `margin = 0.06 + 0.09 × Z` |
| Yapışkan kilit | Uydurma başarısız olsa da son düzlem 8 kare korunur |
| Fail-safe | Normal yeterince dikey değilse (duvar/tavan) veya iç-nokta azsa ayıklama uygulanmaz |

**Neden yüzdelik eşik:** Sabit 6 cm bandı, derinlik ölçeği değiştiğinde (model çözünürlüğü büyüyünce değerler 0.6–2.2 m'den 0.9–4.4 m'ye çıktığında) her turda 100 noktanın altına düşüp uydurmayı tamamen iptal ediyordu — zemin ayıklama sessizce devre dışı kalıyordu.

**Neden mesafeyle uyarlanan marj:** Derinlik gürültüsü uzakta büyür (ölçüldü: 1 m'de ±0.11 m, 2 m'de ±0.21 m). Sabit marj uzakta yetersiz kalıp zemini engel diye sızdırıyordu.

**Neden yapışkan kilit:** Kilit açılıp kapandığında engeller titriyordu.

**Sonuç:** 1 sahte tam-genişlik engel → 5 ayrı gerçek engel. Kamera yüksekliği doğru bulunuyor (sentetik testte tam 0.50 m).

---

### `edge_vision/obstacles.py` — Engel çıkarımı

**`Obstacle`** (dataclass) — Tek bir engel: mesafe, ileri/yanal bileşen, yön açısı, genişlik, bölge (left/center/right), piksel aralığı.

**`ObstacleReport`** (dataclass) — Bir karenin tüm çıktısı: engel listesi, en yakın engel, ham tarama dizileri, **tanımalar** (`detections`) ve **en yakın adlandırılmış nesne** (`nearest_named`).

İki katman birbirini doğrular ama birbirinin yerine geçmez: engel katmanı **her** yüzeyi görür (duvar, kutu, tanımadığı cisim), tanıma katmanı yalnızca COCO'daki 80 sınıfı görür ama onlara **ad** verir. Kaçınma kararları hâlâ engel katmanından çıkar — tanıma kaybolduğunda sistem kör kalmaz.

**`ObstacleScanner`**

| Metot | Görev |
|---|---|
| `scan(depth_m, roi_top, ignore_mask)` | Derinlik haritasını sözde-LiDAR taramasına çevirir |
| `cluster(u, x, z, bearing)` | Komşu sütunları mesafe sürekliliğine göre engellere kümeler |

**Robust sütun taraması:** Saf `argmin` (tek en yakın piksel) yerine **en yakın 5 pikselin ortalaması + en az 3 destek pikseli** şartı. Tek gürültülü piksel sahte yakın engel üretemez — gürültü uzakta belirginleştiği için bu özellikle uzak mesafede önemlidir.

**Kümeleme kuralı:** Derinlik sıçraması mutlak 0.3 m veya mesafenin %15'ini aşarsa yeni küme açılır. 3 sütundan küçük kümeler gürültü sayılıp atılır.

---

### `edge_vision/scan_smoother.py` — Zamansal yumuşatma

**`ScanSmoother`** — Hızlı-yaklaşma / yavaş-uzaklaşma histerezisi.

**Güvenlik felsefesi:** Bir sütunun mesafesi öncekinden **daha yakın** geldiyse değişiklik **anında** kabul edilir — tehlikenin yaklaştığını anlamak asla geciktirilmez. Mesafe **daha uzak** görünüyorsa (nesne kayboldu/uzaklaştı), bu yalnızca `rise_persist_frames` kare boyunca ardışık tutarlı olursa kabul edilir. Tek karelik okluzyon/gürültü sıçraması sistemi yanıltamaz.

---

### `edge_vision/avoidance.py` — Kaçınma katmanı

**`AvoidanceCommand`** (dataclass) — Hareket kontrolcüsünün tüketeceği standart komut: eylem, yön, hız çarpanı, en yakın mesafe, gerekçe.

**`AvoidancePlanner`** — "En geniş açık boşluğu izle" (follow-the-gap) politikası:

| Durum | Koşul | Davranış |
|---|---|---|
| `STOP` | En yakın ≤ `stop_distance_m` | Dur |
| `FOLLOW_GAP` | En yakın ≤ `brake_distance_m` | Yavaşla + boşluğa yönel |
| `CAUTION` | En yakın ≤ `warn_distance_m` | Mesafeyle orantılı yavaşla |
| `CLEAR` | Yol açık | Tam hız |

Bu modül aracı **sürmez**; sadece komut üretir. Gerçek robotta kendi planlayıcınızla değiştirilebilir — `ObstacleReport` tüm ham veriyi zaten taşır.

---

### `edge_vision/depth_color.py` — Renk eşlemesi (TEK KAYNAK)

Renk ↔ mesafe ilişkisinin tek tanımıdır. Derinlik haritası ve engel işaretleri renklerini buradan alır — böylece bir renk **nerede görünürse görünsün aynı mesafeyi** ifade eder.

**İki parçalı ölçek:**

| Aralık | Renk davranışı |
|---|---|
| `viz_near_m` → `viz_red_max_m` (0.05 → 0.40 m) | **Tamamen kırmızı**, doğrusal kademe (0.05 en koyu) |
| `viz_red_max_m` → `viz_far_m` (0.40 → 3.0 m) | Turuncu → sarı → yeşil → mavi (karekök) |

**Ölçek SABİTTİR, sahneye göre kaymaz.** Kayan bir ölçekte aynı renk her karede başka mesafe demek olur ve sisteme dayanan kararlar sürekli hata üretir.

**Palet detayı (ölçüldü):** TURBO paletinin uçları karanlıktır — `dn=255` koyu bordo (R=122), `dn=0` koyu mor. Bu yüzden en yakın cisim "kırmızı" değil koyu görünüyordu. Ölçek paletin canlı bölgesine (`dn ∈ [22, 228]`) sıkıştırıldı; net kırmızı sınırı `dn ≥ 210`.

**`LUT` (256 girişlik önceden hesaplanmış tablo) — kritik performans düzeltmesi:** `applyColorMap`'i 1×1 dizide çağırmak ~0.83 ms sürüyor. Nokta başına çağrıldığında kare başına **~131 ms** yanıyordu ve pencere modu 30 yerine **~7 FPS**'te koşuyordu. Tablo aramasıyla aynı renk mikrosaniyede bulunur; üretilen renkler birebir aynıdır.

| | Önce | Sonra |
|---|---|---|
| `color_at` | 0.83 ms | **0.019 ms** |
| `compose_view` | 141 ms | **4.09 ms** |

---

### `edge_vision/viz.py` — Görselleştirme

| Fonksiyon | Görev |
|---|---|
| `compose_view()` | Yan yana iki panel üretir (kamera \| derinlik veya kenar) |
| `draw_debug()` | Sol panel: nesne kutuları/maskeleri + durum satırları |
| `_draw_detections()` | Nesne kutuları/maskeleri + "EN YAKIN NESNE" bandı |
| `edge_view()` | Kenar haritası paneli |

**Tanıma çizimi:** Tüm kutular soluk gri, **en yakın nesne sarı** ve kalın; adı+mesafesi üstteki bantta da yazılır. Maske varsa cismin kendi sınırı çizilir, kutu yalnızca çerçevedir. Kutu görüntünün üstüne yapışıksa etiket kutunun içine alınır — dışarıda kalırsa üstteki HUD satırlarının arkasında kaybolur (ölçüldü: "insan" etiketi komut satırının altında okunmuyordu).

Tanınan nesneler arka plan bulanıklaştırmasında **ön plan** sayılır: adı okunan cisim bulanık kalmamalı.

### Ekrandan kaldırılan iki katman

Sol panelde başlangıçta iki ayrı mesafe göstergesi vardı; ikisi de kaldırıldı.

**Kuşbakışı radar paneli** sözde-LiDAR verisini (X, Z) düzleminde gösteriyordu — doğru bir tercihti ama sol görüntünün **%42 × %50'sini** kapatıyordu.

**Engel yön iğneleri** her engel sütunu için ayrı bir mesafe etiketi çiziyordu; ekranın ortasında sağa sola koşan bir sayı kalabalığı oluşuyor, görüntüyü okumayı zorlaştırıyordu.

Taşıdıkları bilgi kaybolmadı, zaten başka yerde yazılı:

| Bilgi | Nerede okunuyor |
|---|---|
| En yakın nesnenin adı + mesafesi | Üstteki bant (`EN YAKIN NESNE: sandalye 0.75 m`) |
| Her nesnenin adı + mesafesi | Kutu etiketi (`insan 1.00m`) |
| Cismin sınırı | Segmentasyon maskesinin konturu |
| Engel kararı ve yön | Üst satır (`STOP steer=+0 v=0.00`) |
| Ham engel verisi (yön, mesafe dizileri) | `ObstacleReport` → `--json` köprüsü |

Radarın kaldırılmasıyla `depth_color.colors_for()` kullanılmaz hale geldi ve silindi (tarama nokta bulutunu yalnızca radar renklendiriyordu).

---

### `edge_vision/calib_overlay.py` — Pencere-içi kalibrasyon

**`CalibOverlay`** — Terminal gerektirmez; tüm etkileşim OpenCV penceresindedir.

| Tuş | İşlev |
|---|---|
| `c` / `ESC` | Kalibrasyon modunu aç/kapa |
| `n` | **Sonraki nesne** (adaylar arasında gez) |
| `m` | **Nişan kutusu** (manuel) moduna geç — COCO dışı cisimler için |
| `+` / `-` | Nişan kutusunu büyüt / küçült (%10–%90) |
| rakam, `.` | Gerçek mesafeyi yaz |
| `Backspace` | Sil |
| `ENTER` | Ölçümü kaydet (12 kare toplanır) |
| `f` | Uygula + config'e yaz |
| `x` | Noktaları temizle |
| `r` | Başlangıç kalibrasyonuna dön |

**Hedef seçimi — kilitli:** Ölçülecek cisim sabit bir nişan kutusundan değil, **tanınan nesneler** arasından seçilir ve **kilitlenir**.

| Kural | Neden |
|---|---|
| Hedef iz kimliğine (`tid`) kilitlenir | Her karede yeniden karar verilirse sahneye yeni nesne girdiğinde hedef kendiliğinden ona atlıyor, ölçümün ortasında cisim değişiyordu. İzleyici aynı cismi kareler arası taşıdığı için kilit kutu titremesinden etkilenmez |
| Kilit yalnızca `n` ile veya hedef sahneden çıkınca değişir | Ölçüm boyunca hedefin sabit kalması şart |
| Otomatik seçim = **en büyük** aday | En çok piksel = en güvenilir medyan |
| `min_target_area_frac` (%2) altındaki kutular aday **değil** | Ölçüldü: kapı kolu kutusu karenin **%0.42**'si (323 piksel). Zemin ve arka plan ayıklandıktan sonra güvenilir medyan alacak piksel kalmıyor, ölçüm sessizce yanlış çıkıyordu |

Küçük kutular yine **çizilir** ("kucuk" etiketiyle, kırmızımsı): sistemin o cismi gördüğü ama hedef saymadığı ekranda görünür — yoksa "neden kapı kolunu seçti / seçmiyor" sorusu cevapsız kalır. Üst şeritte kaç adayın atlandığı yazar.

**COCO'da olmayan cisimler (kapı, duvar, karton):** `m` ile nişan kutusuna geçilir ve `+` / `-` ile kutu cismin üstüne **boyutlandırılır** (%10–%90). Kapı gibi büyük ve COCO'da bulunmayan yüzeyler böyle kalibre edilir; kalibrasyon 80 sınıfa mahkûm değildir.

**`measure_object()` — cismi ayıklama:** Hedefin **en ön yüzeyi** bağlı bileşen analiziyle (`connectedComponents`) ayrılır, parlak sarı konturla çizilir; geri kalan her şey bulanıklaştırılıp karartılır. Kontur = ölçülen piksel kümesi; kontur taşıyorsa ölçüm de taşmıştır, kullanıcı bunu gözle görür.

**Segmentasyon maskesi varsa tahmin yoktur:** hedefin maskesi doğrudan ölçüm kümesidir; zemin filtresi bu yolda **uygulanmaz** (bkz. `fuse_distance`). Aşağıdaki bölge büyütme, maskesi olmayan modeller (`yolo11n`) ve nişan kutusu için kalır.

### "Sistem yalnızca yüzümü ölçüyor" — iki ayrı sebep

Aynı belirtinin iki farklı kaynağı ölçülerek ayrıştırıldı:

**1. Zemin maskesi cismi yiyordu.** Zemin düzlemi kilitliyken bir insanın maskesinin **%63'ü** zemin sayılıp atılıyor, geriye üst gövde kalıyordu. Düzlem uydurma iyi olmadığında (iç-nokta 0.18–0.45) gövde "zeminin altında" görünüyor. Çözüm: maske yolunda zemin filtresi kaldırıldı — ölçülen maske %50 → **%100**.

**2. Nişan kutusu bölgeyi kırpıyordu.** Manuel kipte büyüme kutunun dışına **çıkamaz**. Ölçüldü:

| Bölge büyütme eşiği | Kişinin yakalanan oranı |
|---|---|
| 0.30 (varsayılan) | %18.6 |
| 0.60 (iki katı) | %19.9 |

Eşiği iki katına çıkarmak yalnızca 1.3 puan kazandırıyor — **sınır eşik değil, kutunun kendisi**:

| Nişan kutusu | Yakalanan oran |
|---|---|
| %28 (varsayılan) | %7.4 |
| %52 | %30.0 |
| %80 | %68.5 |

Bu yüzden hassasiyet eşiği yükseltilmedi (yükseltmek yalnızca arka plana sızma riskini artırırdı). Bunun yerine:

* `c` ile kalibrasyona her girişte **nesne kipi** varsayılan oldu. `m` yapışkandı; bir kez kapı ölçmek için manuel kipe geçen kullanıcı sonraki oturumlarda farkında olmadan kutuyla ölçmeye devam ediyordu.
* Bölge kutunun kenarına dayandığında ekranda uyarı çıkıyor: `nisan kutusu %28  <- KUTU KESIYOR, + ile buyut`.

**Ayırma ölçütü mutlak mesafe değil, derinlik SÜREKLİLİĞİDİR.** Sabit bant (`|d − ön_yüzey| ≤ marj`) cismi mesafeye göre keser: yüz 0.45 m, üstteki forma 0.60 m ise ±8 cm'lik bant ikisini ayırır ve sistem **tek bir insanı iki cisim gibi** gösterir. Bölge büyütme (flood fill) komşu piksel farkına bakar: sürekli yüzey boyunca yayılır, arka plana sıçramada durur.

Büyüme **logaritmik derinlikte** yapılır, böylece eşik oransal olur:

| Geçiş | Mutlak | Log uzayında |
|---|---|---|
| Yüz 0.45 → forma 0.60 m | 0.15 m | **0.29** |
| Forma 0.60 → arka plan 2.5 m | 1.90 m | **1.43** |

Aradaki 5 kat fark tek eşikle (`grow_log_tol` = 0.30) ayrılır ve eşik her mesafede aynı anlama gelir.

| Kısıt | Neden |
|---|---|
| **Zemin dışlanır** (`ground_mask`) | Yer, cismi arka planla köprüleyen ana yoldur: cismin dibindeki zemin cisimle aynı mesafededir, büyüme oradan tüm odaya yayılıyordu |
| **Büyüme hedef kutusuyla sınırlı** | Benzer mesafedeki komşu eşya ölçümü kendine çekemez |
| **Yedek bant yöntemi** | Büyüme tutmazsa (parçalı/gürültülü yüzey) eski yöntem devreye girer |

**Ölçülen etki** (sentetik: yüz 0.45 m + forma 0.60 m, arka plan 2.5 m):

| Geçiş genişliği | Bant | Bölge büyütme |
|---|---|---|
| Yumuşak (13 cm rampa) | %88.6 | **%100** |
| Sert basamak, 6 px | %51.7 | **%100** |
| Sert basamak, 1 px | %50.5 | **%100** |

Arka plana sızma: **0 piksel**. Canlı sahnede hedef kutusunun doluluğu %21–41'den **%64'e** çıktı.

**Ölçülen etki** (aynı sahne, sandalye 0.49 m):

| | Maskenin kapladığı kare oranı |
|---|---|
| Eski yöntem (`input_size` 210, tüm kare taranıyor, zemin dahil) | **%88.7** (max %95) — pratikte "tek cisim" = tüm sahne |
| Yalnızca çözünürlük düzeltmesi (518) | %25.0 |
| Çözünürlük + zemin + pencere + kutu kısıtı | **%12.4** (hedef kutusunun %41'i = cismin kendi silüeti) |

**Neden düz medyan değil:** Cisim kutuyu doldurmuyorsa düz medyan arka planla karışır. Sentetik testte: eski düz medyan **3.00 m** (arka plan!), yeni yöntem **0.60 m** (doğru).

**Neden noktalar HAM değerden alınır:** Kalibre değerden alınsaydı her kalibrasyon bir öncekinin üstüne biner, noktalar farklı referans çerçevelerine dağılır ve arka arkaya kalibrasyon sistemi bozardı. Ham değer mutlak olduğu için noktalar biriktirilebilir ve `f`'ye tekrar basmak zararsızdır.

**Güvenlik kontrolleri:**

| Kontrol | Amaç |
|---|---|
| Kararsız ölçüm reddi | 12 karenin sapması yüksekse nokta kaydedilmez |
| Monotonluk | Ham artarken gerçek mesafe de artmalı (yoksa model o aralığı ayırt edemiyor) |
| Aynı ham değer | İki farklı mesafede aynı ham → model doygunlukta |
| **Aralık genişliği** | `span < 0.30 m` **ve** `oran < 2.0` ise reddet |
| **Eğim dirseği** | Yakın parça uzak parçanın `3x`'inden dikse reddet (asimetrik) |
| **Tek başına dik parça** | Bir parça tablonun genel eğiminin `3x`'ini aşarsa reddet |
| **Mesafe sınırı** | `0.02 m` – `20 m` dışı reddedilir; santim uyarısı verilir |
| Kapsama bildirimi | Uygulandığında kapsanan aralık yazılır, 0.5 m altı nokta yoksa uyarı |

**Aralık genişliği kontrolü neden kritik:** Gerçek bir vakada kullanıcı 0.61 m ve 0.70 m'de (yalnızca 9 cm arayla) iki nokta girdi. Bu iki noktanın eğimi (0.168) ekstrapolasyonla **tüm menzili** yönetince sahnenin tamamı 0.49–1.08 m'ye sıkıştı ve derinlik haritasından mavi tamamen kayboldu.

### Uzak menzil — tablonun kapsamadığı yer tahmindir

Ölçüldü (kamera odanın uzak ucuna doğrultulmuş, 209 derinlik haritası):

| | Ham değer |
|---|---|
| Sahne ortancası | 2.98 |
| Sahne tavanı | **3.37 – 3.49** |
| 5 cm'deki cisim | 0.38 |

**Model uzakta doymuyor** — kullanılabilir ham aralığı 0.38 ≈ 3.4. Ama 0.05–0.61 m arası kalibre edilmiş bir tablo bu aralığın yalnızca **%39'unu** (0.38 → 1.55) kapsar; kalan **%61'i** son parçanın eğimiyle tahmin edilir. Sonuç: 4 m'deki duvar 1.48 m okunur. Bu modelin körlüğü değil, **kalibrasyon kapsamı** sorunudur.

Tek bir uzak nokta eklemenin etkisi (aynı tablo + 4.00 m @ ham 3.37):

| Ham | Uzak nokta YOK | Uzak nokta VAR |
|---|---|---|
| 2.00 | 0.82 m | **1.45 m** |
| 2.98 | 1.29 m | **3.27 m** |
| 3.37 | 1.48 m | **4.00 m** |

Uzak nokta almanın pratik yolu: kamerayı karşı duvara doğrultun, `m` ile nişan kutusuna geçip `+` ile kutuyu duvarın üstüne büyütün, mesafeyi metreyle ölçüp girin.

### Eğim dirseği kontrolü — okumanın zıplaması

Gerçek bir vakada 0.20 / 0.70 / 1.00 m ölçülüp uygulandı:

| Parça | Ham aralık | Eğim | Ham gürültünün (σ=0.0156) mesafeye yansıması |
|---|---|---|---|
| 0.20–0.70 m | 0.4291 → 0.6233 (**0.19**) | 2.575 | **±4.0 cm** |
| 0.70–1.00 m | 0.6233 → 1.3503 (0.73) | 0.413 | ±0.6 cm |

Komşu parçalar arasında **6.2x eğim sıçraması**. Sonuç: dirseğin iki yakasında rejim değiştiği için okuma 0.45 ile 0.68 arasında zıpladı.

**Teşhis:** İki uç nokta birbiriyle tutarlıydı (tek eğim 0.868). Bu eğime göre 0.70 m'nin ham değeri ~1.00 olmalıydı; ölçülen 0.6233 ise ~0.37 m'ye karşılık geliyor — yani **ölçüm hedefin önündeki bir yüzeyi yakalamış**, model doymamış. Orta nokta çıkarıldı, tablo tek eğime döndü (her yerde ±1.4 cm).

**Kural — ASİMETRİK:** Yalnızca eğimin mesafeyle **azaldığı** durum reddedilir (`yakın parça > 3x uzak parça`). Tersi serbesttir.

**Neden asimetrik:** Monoküler derinlik modeli uzağı sıkıştırır, yani metre/ham birim mesafeyle **artar**. Ölçüldü: 0.05–0.61 m yalnızca 1.17 ham birim kaplarken 0.61 m'nin ötesi 1.82 birim kaplıyor. Dolayısıyla doğru bir uzak nokta eklendiğinde son parça komşusundan birkaç kat dik olur (örnekte 1.86 / 0.37 = 5x) — **simetrik kontrol doğru kalibrasyonu reddederdi.** Tersi ise fizik dışıdır ve hatalı ölçümün imzasıdır.

| Tablo | Eğimler | Düşen dirsek | Sonuç |
|---|---|---|---|
| Bozuk (0.20/0.70/1.00) | 2.57, 0.41 | 6.24x | **RED** |
| Mevcut (0.05/0.30/0.61) | 0.74, 0.37 | 2.01x | GEÇER |
| Mevcut + uzak nokta (4.00 m) | 0.74, 0.37, **1.86** | 2.01x | GEÇER |

Uyarı `f`'ye basmadan **önce** de ekranda görünür (`dirsek 0.20-0.70m 6.2x ZIPLAR!`), böylece hangi noktanın bozuk olduğu ölçüm sırasında fark edilir.

### Tek başına dik parça — asimetrik kontrolün ikinci açığı

Dirsek kontrolü yalnızca **komşulara** bakar ve eğimin mesafeyle artmasını normal sayar. Ama iki nokta ham eksende birbirine çok yakınsa o parça tek başına uçar ve komşu kıyaslaması bunu yakalayamaz.

Gerçek vaka: `0.70 m` ve `1.40 m` noktaları yalnızca **0.044 ham birim** arayla ölçüldü.

| Parça | Ham aralık | Eğim | σ=0.0156'nın yansıması |
|---|---|---|---|
| 0.20–0.70 m | 0.6330 → 1.0067 (0.374) | 1.34 | ±2.1 cm |
| 0.70–1.40 m | 1.0067 → 1.0510 (**0.044**) | **15.8** | **±24.7 cm** |

Tablonun geneli 2.87 eğimindeyken bu parça **5.5 katı** dik. Sonuç: okuma yerinde durmuyor, kalibrasyon sırasında değerler okunamayacak hızda değişiyor. Ayrıca bu tablo, modelin küresel ölçek kaymasını da büyütüyordu — ham eksendeki %50'lik kayma ekranda %85'e çıkıyordu.

**Kural:** Hiçbir parça tablonun genel eğiminin `MAX_STEEP` (3x) katından dik olamaz.

| Tablo | En dik parça | Genel eğime oranı | Gürültü | Sonuç |
|---|---|---|---|---|
| 0.20/0.70/1.40 | 15.80 | **5.50x** | ±24.7 cm | **RED** |
| 0.05/0.30/0.61/2.10 | 1.05 | 1.35x | ±1.6 cm | GEÇER |

### Santim/metre karışıklığı — asimetrik kontrolün açığı

Gerçek vaka: kullanıcı 2.10 m yerine **"210"** yazdı (santim) ve tabloya **210 metrelik** bir nokta girdi. Hiçbir mevcut kontrol bunu yakalamadı:

| Kontrol | Neden geçti |
|---|---|
| Monotonluk | 210 > 0.61 ✓ |
| Aralık genişliği | span devasa ✓ |
| **Eğim dirseği** | Uzak parça 147.9 eğim verdi — ama **asimetrik** kontrol "uzak parça daha dik olabilir" diyor (fizik gereği doğru), bu yüzden geçti |

Asimetrik kontrol doğru bir karardı ama tek başına yetmiyor: "makul ölçüde dik" ile "absürt derecede dik" arasını ayıramaz. Çözüm ölçüt sıkılaştırmak değil, **girdiyi sınırlamak**:

* `_start_capture()` — `0.02 m` altı / `20 m` üstü reddedilir ve doğrudan öneri verilir: *"210 m fazla buyuk — SANTIM mi yazdiniz? 2.10 m icin '2.10' yazin"*.
* `_apply()` — config'ten elle girilmiş ya da eski sürümden kalmış imkânsız noktalar da durdurulur; giriş kontrolü tek başına yeterli değildir.

### Ölçüm hassasiyeti

| Ayar | Önce | Sonra | Etki |
|---|---|---|---|
| `capture_frames` | 12 | **16** | Medyanın standart hatası **%12.5** düştü (20000 denemede ölçüldü; teorik `1−√(12/16)` = %13.4). Nokta başına ~0.2 s ek maliyet |
| `band_tol` | 0.15 | **0.13** | Ayırma bandı 0.50 m'lik cisimde ±9.5 → **±8.5 cm** |
| `stability_tol` | 0.20 | **0.17** | Kararsız ölçüm daha erken reddedilir |

Kasıtlı olarak küçük bir artış: bant çok daraltılırsa cisim parçalanır, kararlılık eşiği çok sıkılırsa hiçbir ölçüm kabul edilmez.

---

## 4. Kalibrasyon Sistemi (Detaylı)

### Ham değer vs kalibre değer

| | Ham | Kalibre |
|---|---|---|
| Nedir | Modelin doğrudan çıktısı | Ham'a dönüşüm uygulanmış hali |
| Uzakta | İsabetli (model eğitim alanında) | Kalibrasyona bağlı |
| Çok yakında (5 cm) | Yanlış (~0.75) | Düzeltilmiş |
| Kim kullanır | Yalnızca kalibrasyon ölçümleri | Ekran, engel kararları |

Model **oda ölçeğindeki iç mekan** verisiyle eğitildi. Duvar, sandalye, arka plan bu aralıkta olduğu için ham değer uzakta doğal olarak tutar. 5 cm ise modelin hiç görmediği bir durumdur — orada şaşırır.

### İki kalibrasyon modu

**1. Afin (varsayılan, `calib_points` boşken):**
```
gerçek = calib_scale × ham + calib_offset
```

**2. Parçalı tablo (`calib_points` ≥ 2 nokta):**
Noktalar arası doğrusal interpolasyon; menzil dışında **tablonun genel eğimiyle** ekstrapolasyon.

### Neden tek doğru yetmez

Modelin ham çıktısı ile gerçek mesafe arasındaki ilişki **doğrusal değildir**. Yakında ham değer sıkışır, oda ölçeğinde açılır. Tek bir afin doğru seçilirse ya yakın ya uzak doğru olur, ikisi birden olmaz.

Sentetik doğrulama:

| Yöntem | En büyük hata |
|---|---|
| Tek doğru (afin), en iyi uyum | **0.096 m** |
| Parçalı tablo | **0.0000 m** |

### Ekstrapolasyon neden genel eğimle

Son iki nokta birbirine yakınsa o parçanın eğimi ölçüm gürültüsüne boğulur — ama ekstrapolasyon tüm menzili yönetir. Tablonun genel eğimi `(ys[-1]−ys[0]) / (xs[-1]−xs[0])` tüm noktaları kullandığı için tek bir gürültülü çifte teslim olmaz.

### Doğru kalibrasyon tarifi

`c` ile kalibrasyona girin. Her nokta için:

1. Cismi kameranın önüne koyun — **tanınan bir nesne** (şişe, dizüstü, sandalye, insan) hedef seçimini kolaylaştırır, ama şart değildir. Sistem en büyük adaya kilitlenir ve ölçüm boyunca onu bırakmaz.
2. Hedef doğru cisim değilse `n` ile aday değiştirin. Cisim tanınmıyorsa (kapı, duvar) `m` ile nişan kutusuna geçip `+`/`-` ile kutuyu cismin üstüne boyutlandırın.
3. **Konturun cismi sardığını** gözle doğrulayın — kontur arka plana taşıyorsa ölçüm de taşar, o noktayı kaydetmeyin.
4. Gerçek mesafeyi (cetvelle ölçülmüş) yazın, `ENTER` ile 12 kare toplayın.
5. Tüm noktalar bittiğinde `f` ile uygulayın.

Mesafeler **birbirinden uzak** olmalı:

| Nokta | Öneri | Ham karşılığı (bu makinede ölçüldü) |
|---|---|---|
| 1 | 0.05 m (el mercekte) | ~0.38 |
| 2 | 0.30 m | ~0.72 |
| 3 | 1.0 m | — |
| 4 | 2.0 m | — |
| 5 | 3–4 m (karşı duvar) | ~3.0–3.4 |

**Uzak noktayı atlamayın.** Ham aralığın %61'i 1.55'in ötesindedir; oraya nokta konmazsa odanın tamamı birkaç on santime sıkışır.

Yakın noktaları atlamayın — atlanırsa yakın davranış ekstrapolasyona düşer. Uzak noktayı atlamayın — atlanırsa mavi renk hiç çıkmaz.

---

## 5. Yapılandırma Referansı (`config.yaml`)

### `camera`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `source` | `0` | Kamera indeksi veya video yolu |
| `capture_width` / `capture_height` | 640 / 480 | Yakalama çözünürlüğü |
| `fps` | 30 | İstenen kare hızı |
| `hfov_deg` | 62.2 | Yatay görüş açısı (yanal konum için) |

### `processing`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `proc_width` | 320 | İşleme genişliği (en-boy oranı korunur) |
| `roi_top_ratio` | 0.30 | Bu oranın üstü taranmaz (tavan/gökyüzü) |
| `column_stride` | 2 | Tarama sütun adımı |

### `depth`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `backend` | `dav2` | `dav2` (kare girdi + ImageNet normalizasyonu) veya `yolo26` (4:3 girdi, yalnızca /255) |
| `model` | ONNX yolu | Depth Anything V2 metrik model |
| `input_size` | 518 | 14'ün katı olmalı (ViT yaması); modelin yerleşik çözünürlüğü |
| `providers` | DirectML, CPU | Denenecek ONNX sağlayıcıları |
| `threads` | 4 | CPU iş parçacığı sayısı |
| `median_ksize` | 3 | Uzamsal median (0=kapalı, 3 veya 5) |
| `calib_scale` / `calib_offset` | 0.514 / −0.147 | Kaba afin başlangıç (tablo boşken kullanılır) |
| `min_valid_m` | 0.05 | En yakın okuma tabanı |
| `calib_points` | `[]` | Parçalı kalibrasyon tablosu — `c` ile doldurulur |
| `viz_red_max_m` | 0.40 | Kırmızı bandın üst sınırı |
| `viz_near_m` / `viz_far_m` | 0.05 / 3.0 | Renk skalası uçları |

### Model seçenekleri (ölçüldü)

Üç model bu makinede aynı sahnede karşılaştırıldı. Hepsi `models/` altında hazır.

**Tanıma** (40 kare, aynı görüntü):

| Model | Süre (12 iş parçacığı) | Maske | Bulduğu |
|---|---|---|---|
| `yolo11n` (kutu, COCO80) | 44.7 ms | — | insan 33, **kedi 6** (hayalet) |
| **`yolo26n-seg`** (varsayılan) | 61.3 ms | %79 | insan 50, televizyon 31 |
| `yoloe-26n-seg` (16 özel sınıf) | 58.8 ms | %79 | insan 41, sandalye 9, televizyon 7 |

`yolo11n` sahnede olmayan bir kedi uydurdu; YOLO26 modelleri uydurmadı ve televizyonu buldu. **`yoloe-26n-seg`** açık sözlüklüdür: `kapı`, `duvar`, `pencere`, `merdiven`, `dolap` gibi **COCO'da olmayan** sınıfları metin komutuyla tanır — kapıyı nişan kutusuyla elle çerçeveleme derdini kaldırır. Geçiş için `config.yaml` → `yolo.model` + `yolo.classes`.

**Derinlik** (20 kare, aynı sahne):

| `backend` | Model | DirectML | Ayrım (p98/p2) | En yakın okuma std |
|---|---|---|---|---|
| **`dav2`** (varsayılan) | Depth Anything V2, 518 | 40 ms | **4.04x** | 1.10 cm |
| `yolo26` | `yolo26n-depth`, 480×640 | **6.6 ms** | 2.84x | 1.47 cm |

`yolo26` **6 kat hızlıdır** ve doğrudan metre üretir (kalibrasyon gerektirmez); boru hattında ölçüldü: derinlik tazeleme 10.9 → **30.6 FPS**, yani harita artık kamera hızında yenileniyor ve darboğaz olmaktan çıkıyor. Buna karşılık bu sahnede dinamik aralığı daha dardı.

**Hangisi doğru ölçüyor — `compare_depth.py`:** İki motoru metre kullanmadan ayırt eder. Sistem zemini bir düzlem olarak uyduruyor ve oradan kameranın yerden yüksekliğini çıkarıyor; bir modelin ölçeği k katı yanlışsa bulduğu yükseklik de k katı yanlış çıkar. Kameranızın yüksekliğini zaten bildiğiniz için bu tek sayı karar vermeye yeter.

```bash
python compare_depth.py
```

Kalibrasyon düzeltildikten sonraki ölçüm:

| backend | Çıkarım | Tazeleme | Zemin kilidi | **Kamera yüksekliği** |
|---|---|---|---|---|
| `dav2` | 83.2 ms | 11.5 fps | %100 | **0.35 m** |
| `yolo26` | **11.9 ms** | **31.1 fps** | %100 | **0.36 m** |

**Oran 0.99x** — biri kalibrasyonla, diğeri hiç kalibrasyon olmadan, iki bağımsız model aynı metrik ölçeğe %1 içinde ulaştı. Bu, kalibrasyon tablosunun **doğru** olduğunun karşılıklı doğrulamasıdır (yolo26 kalibrasyon kullanmaz; dav2 ona uyuyorsa tablo tutarlıdır).

`yolo26` tarafında harita **kamera hızında** (31 fps) yenileniyor — derinlik artık darboğaz değil.

**Ham değerleri ortak değildir** — `backend` değiştirilirse `calib_points` geçersizdir.

### `yolo`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `enabled` | true | Tanıma katmanı açık/kapalı (kapalıysa sistem eskisi gibi çalışır) |
| `model` | `models/yolo26n-seg.onnx` | YOLO26 nano segmentasyon, COCO 80 sınıf |
| `input_size` | 640 | Model sabit boyutluysa oturumdan okunan değer kazanır |
| `conf_thres` | 0.35 | Bu güvenin altındaki tespitler atılır |
| `iou_thres` | 0.45 | NMS örtüşme eşiği |
| `max_det` | 20 | Kare başına en fazla tespit |
| `providers` | CPU | **CPU zorunlu**: derinlik ağı DirectML'i tutuyor |
| `threads` | 12 | CPU iş parçacığı. Ölçüldü (yolo26n-seg): 2→103 ms, 4→92, 6→70, 8→62, **12→53** |
| `classes` | `[]` | Açık sözlüklü model (YOLOE) kullanılıyorsa sınıf adları; boşsa COCO listesi |

### `calib`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `min_target_area_frac` | 0.02 | Kare alanının bu oranından küçük kutular hedef sayılmaz |
| `aim_frac` | 0.28 | Nişan kutusunun kenar oranı (`+`/`-` ile değişir) |
| `aim_min` / `aim_max` | 0.10 / 0.90 | Nişan kutusu sınırları |
| `aim_step` | 0.04 | Tuş başına değişim |
| `capture_frames` | 16 | Nokta başına toplanan kare (medyanın hatası ~σ/√n) |
| `grow_log_tol` | 0.30 | Bölge büyütme adım eşiği (log derinlik = oransal). Büyütürsen cisim arka planla birleşir, küçültürsen cisim parçalanır |
| `band_tol` | 0.13 | Yedek bant yöntemi için derinlik bandı (±%) |
| `stability_tol` | 0.17 | Ölçüm sapması bu oranı aşarsa nokta reddedilir |

### `stabilize`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `scale_anchor` | true | Küresel ölçek kayması düzeltmesi |
| `ref_pct` | 85 | "Uzak arka plan" olarak alınan yüzdelik |
| `warmup_frames` | 20 | Referans bu kadar kare sonra kilitlenir |
| `anchor_min` / `anchor_max` | 0.30 / 3.00 | Düzeltme katsayısı sınırları |
| `anchor_ema` | 0.25 | Ani sıçrama yerine yumuşak düzeltme |

### `hand`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `enabled` | true | Paket/model yoksa kendiliğinden kapanır |
| `model` | `models/hand_landmarker.task` | MediaPipe modeli (~7.5 MB) |
| `num_hands` | 2 | En fazla kaç el |
| `conf_thres` | 0.5 | MediaPipe tanıma eşiği |
| `box_pad` | 0.12 | Eklem noktaları eti kapsamaz, kutu genişletilir |

### `tracking`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `enabled` | true | Kapalıysa ham YOLO çıktısı doğrudan gösterilir (etiket titrer) |
| `vote_frames` | 15 | Etiket son N karenin çoğunluğu (~0.75 s @ 20 fps) |
| `min_hits` | 3 | Bu kadar kare görülmeden ekrana gelmez |
| `max_miss` | 5 | Kısa kayboluşta iz silinmez |
| `iou_match` | 0.30 | Kutu eşleme eşiği |
| `switch_ratio` | 1.5 | Etiket değişimi için gereken oy üstünlüğü |
| `box_ema` | 0.5 | Kutu yumuşatma |
| `deadband_m` | 0.06 | Bu farkın altındaki oynama bastırılır |
| `alpha_still` / `alpha_move` | 0.12 / 0.60 | Duran / hareket eden cisim takip hızı |

### `avoidance`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `max_range_m` | 5.0 | Ötesi "açık" sayılır |
| `warn_distance_m` | 1.2 | Uyarı bölgesi |
| `brake_distance_m` | 0.6 | Fren bölgesi |
| `stop_distance_m` | 0.3 | Acil duruş |
| `center_zone_deg` | 12.0 | Merkez koridor yarı-açısı |
| `min_gap_deg` | 15.0 | Yönlenilebilir minimum boşluk |
| `max_tracked_objects` | 5 | Yalnızca en yakın N küme engel sayılır |

### `ground`
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `enabled` | true | Zemin ayıklama açık/kapalı |
| `margin_base_m` | 0.06 | Marjın sabit bileşeni |
| `margin_per_m` | 0.09 | Marjın mesafeyle büyüyen bileşeni |
| `min_inlier_ratio` | 0.25 | Altındaysa zemin sayma (fail-safe) |
| `min_normal_y` | 0.80 | Normal bu kadar dikey değilse uygulama |
| `fit_step` | 4 | Uydurma örnekleme adımı |
| `ema` | 0.25 | Düzlem yumuşatma katsayısı |
| `max_miss` | 8 | Kilit koptuktan sonra kaç kare korunur |

---

## 6. Performans Analizi

### Nerede darboğaz var

Ölçüldü:

| Aşama | Süre |
|---|---|
| Kamera ham hızı | 29.8 FPS (**donanımsal tavan**) |
| `pipeline.process` | 1.3 ms |
| `compose_view` | 4.1 ms |
| Derinlik ağı `session.run` | 27.75 ms (toplamın %95'i) |
| Tanıma `session.run` (CPU) | 26.8 ms (ayrı iş parçacığı — ana döngüye yansımaz) |
| Kutu → mesafe füzyonu | < 0.1 ms (kutu başına tek medyan) |
| Ön işleme | 0.74 ms |
| Son işleme + kalibrasyon + median | 0.12 ms |

**Sonuç:** Sistem **kamera-bağımlıdır**, hesap-bağımlı değil. İşleme 33 ms bütçenin yalnızca %15'ini kullanır; **%85 boş kapasite** vardır.

### Kamera tavanı

Her çözünürlükte ~29.8 FPS. MJPG isteği kabul edilmiyor (YUY2'ye düşüyor), 1280×720'de 10 FPS'e iniyor. Ana döngü FPS'i yazılımla artırılamaz.

### Yapılan optimizasyonlar

| Optimizasyon | Kazanç |
|---|---|
| Renk LUT'u | `compose_view` 141 → 4.09 ms |
| Harita sürüm sayacı | %35 daha az CPU (aynı harita için düzlem uydurma tekrarlanmaz) |
| Koşullu kenar hesabı | Gösterilmiyorsa hiç hesaplanmaz |

Sürüm sayacının yan faydası: aynı derinlik karesi zemin EMA'sına iki kez sayılmadığı için engel okuması **daha kararlı** (std 0.061 → 0.019).

---

## 7. Bilinen Sınırlar

| Sınır | Açıklama |
|---|---|
| Kamera 30 FPS | Donanımsal tavan; yazılımla aşılamaz |
| Çok yakın mesafe (< 10 cm) | Web kamerası odaklanamaz; model de eğitim alanı dışında |
| Model doygunluğu | İki farklı mesafede aynı ham değer üretilirse hiçbir kalibrasyon ayıramaz (araç bunu tespit edip uyarır) |
| Zemin ayıklama sahneye bağlı | Kullanıcı çerçeveyi doldurunca zemin görünmez ve kilit açılmaz — bu doğru davranıştır |
| Kalibrasyon kullanıcı ölçümü gerektirir | Gerçek mesafeler cetvelle ölçülmeli; sistem kendi kendini kalibre edemez |
| Tanıma 80 sınıfla sınırlı | COCO dışındaki cisim (duvar, kapı, kablo) **adsız** kalır — ama engel katmanı onu yine görür ve kaçınma çalışır. `yoloe-26n-seg` bu sınırı kaldırır |
| İki derinlik motoru farklı metre veriyor | Aynı kişi için `dav2` 0.36 m, `yolo26` 0.63 m okudu. Hangisinin doğru olduğu **metreyle** doğrulanmalı |
| Nesne mesafesi haritanın doğruluğu kadar | Füzyon kutu içindeki derinliği okur; kalibrasyon bozuksa etiket doğru, metre yanlış olur |

### Girdi çözünürlüğü — neden 518

Ayrım gücü = aynı karede ham değerlerin `p95/p5` oranı. Bu makinede ölçüldü:

| `input_size` | Ayrım gücü | Süre | Ham aralık |
|---|---|---|---|
| 210 | **1.43x** | 32 ms | 1.89 – 2.70 |
| 322 | 2.00x | 59 ms | 1.26 – 2.52 |
| **518** (yerleşik) | **2.26x** | **40 ms** | 1.19 – 2.69 |

210'da tüm sahne 1.89–2.70 aralığına sıkışıyordu: 0.5 m'deki cisim ile arkasındaki duvar neredeyse aynı ham değeri üretiyor, derinlik paneli tek renk çıkıyor ve **kalibrasyon imkânsız hale geliyordu** — aracın kendi kontrolü ("aynı ham değer iki mesafede") noktaları haklı olarak reddediyordu.

518, modelin eğitildiği çözünürlüktür; ölçümde hem en iyi ayrımı verdi hem de 322'den **hızlı** çıktı (DirectML yerel çözünürlükte daha verimli koşuyor). Ana döngü FPS'i değişmedi (20.1 → 20.1): maliyet arka plan iş parçacığında erir.

**Uyarı:** Çözünürlük değişimi ham değer dağılımını tamamen değiştirir ve eski kalibrasyonu geçersiz kılar. 210 için üretilmiş tablo 518'de **zararlıdır** — eski tablonun üçüncü noktası ham 4.11'deydi, 518'de ham değer 2.7'yi aşmadığı için o nokta hiç ulaşılamıyor ve harita 0.7 m'de doyuyordu. Bu yüzden `calib_points` boşaltıldı; `c` ile yeniden doldurulmalıdır.

---



## 8. Kurulum

Depoyu klonladıktan sonra **üç komut** yeter; hiçbir dosya elle taşınmaz, config elle düzenlenmez:

```bash
pip install -r requirements.txt
```

```bash
pip install ultralytics
```

```bash
python setup_models.py --depth yolo26
```

`setup_models.py` eksik ONNX'leri üretip `models/` altına yerleştirir, ara `.pt` dosyalarını siler ve `--depth yolo26` verilirse `config.yaml`'ı da ayarlar (önce `config.yaml.bak` yedeğini alarak). Zaten var olan modele dokunmaz, tekrar tekrar çalıştırılabilir.

Sonra:

```bash
python main.py
```

**Depth Anything V2 kullanmak isterseniz** (daha geniş dinamik aralık, ama 8 kat yavaş): `--depth yolo26` yerine sade `python setup_models.py` çalıştırın. O model ultralytics'in değildir, dışa aktarım komutu yoktur — `models/` altına elle konulmalıdır; betik eksikse ne yapılacağını yazar.

Açık sözlüklü modeli de üretmek için: `python setup_models.py --yoloe`

Model dosyası eksikken sistem ham ONNX hatası yerine ne yapılacağını söyler:

```
FileNotFoundError: Model dosyasi yok: models/yolo26n-seg.onnx
ONNX dosyalari depoya girmez (.gitignore). Uretmek icin:
    pip install ultralytics
    python setup_models.py
```

Windows'ta GPU hızlandırma için ek olarak:
```bash
pip install onnxruntime-directml
```

Model dosyaları pip ile gelmez, `models/` altına elle konur:

| Dosya | Boyut | Görev |
|---|---|---|
| `depth_anything_v2_metric_indoor.onnx` | ~95 MB | Metrik mesafe (varsayılan) |
| `yolo26n-seg.onnx` | ~11 MB | Nesne tanıma + maske (varsayılan) |
| `yolo11n.onnx` | ~10 MB | Kutu-only alternatif |
| `hand_landmarker.task` | ~7.5 MB | El tanıma (MediaPipe, opsiyonel) |
| `yolo26n-depth.onnx` | ~21 MB | Alternatif derinlik motoru |
| `yoloe-26n-seg.onnx` | ~11 MB | Açık sözlüklü (kapı, duvar…) |

ONNX dosyaları depoya girmez (`.gitignore`). Üretmek için (yalnızca dışa aktarımda `ultralytics` gerekir, çalışma zamanında gerekmez):

```bash
yolo export model=yolo26n-seg.pt format=onnx imgsz=640 opset=12
```

```bash
yolo export model=yolo26n-depth.pt format=onnx imgsz=480,640 opset=12
```

Açık sözlüklü model, sözlüğü dışa aktarımda gömer:

```bash
python -c "from ultralytics import YOLOE; L=['person','door','window','wall','stairs','television','chair','couch','dining table','bottle','cup','laptop','backpack','potted plant','refrigerator','cabinet']; m=YOLOE('yoloe-26n-seg.pt'); m.set_classes(L, m.get_text_pe(L)); m.export(format='onnx', imgsz=640, opset=12)"
```

---

## 9. Lisans

Bu proje **GNU AGPL-3.0** ile lisanslanmıştır — tam metin: [LICENSE](LICENSE).

Copyright (C) 2026 Mustafa Yiğit Acar

**Neden AGPL:** Sistem, Ultralytics'in AGPL-3.0 lisanslı modellerini kullanır. Aynı lisansı seçmek uyumu kendiliğinden sağlar; kaynak zaten herkese açıktır.

Bu depoda **model ağırlığı yoktur**; kod çalışma zamanında yalnızca `numpy`, `opencv-python`, `PyYAML` ve `onnxruntime` kullanır. Modeller dışarıdan gelir ve kendi lisanslarını taşır:

| Bileşen | Kaynak | Lisans |
|---|---|---|
| `yolo26n-seg`, `yolo26n-depth`, `yoloe-26n-seg` | Ultralytics | AGPL-3.0 |
| `depth_anything_v2_metric_indoor` | Depth Anything V2 | Modelin kendi lisansı |
| `ultralytics` paketi | yalnızca `setup_models.py` içinde, dışa aktarım için | AGPL-3.0 |

Ticari bir kullanım düşünülürse Ultralytics'in ayrı ticari lisansı gerekir.

---

## 10. Tasarım Kararlarının Özeti

| Karar | Gerekçe |
|---|---|
| Tek mesafe kaynağı (metrik derinlik ağı) | Klasik monoküler geometri yığını kırılgandı ve kalibrasyona bağımlıydı |
| Ayrı iş parçacığında derinlik | Ana döngünün 30 FPS'te kalması için |
| Tanıma ayrı katman | Derinlik "ne kadar uzakta" der, "ne" olduğunu bilmez; kaçınma yine engel katmanından çıkar |
| YOLO CPU'da | İki DirectML oturumu çakışıyor; 27 ms zaten arka planda erir |
| Kutu mesafesi = çekirdek medyanı | Kutu köşeleri arka planı içerir, cisim ortadadır |
| Zemin düzlemi ayıklama | Zemin, sütun taramasını domine edip gerçek cisimleri gizliyordu |
| Parçalı kalibrasyon | Tek doğru yakın ve uzağı aynı anda tutturamaz |
| Kalibrasyon noktaları hamdan | Kalibre değerden alınırsa üst üste kalibrasyonlar birbirini bozar |
| Sabit renk ölçeği | Kayan ölçekte aynı renk her karede başka mesafe demek olur |
| Radar paneli kaldırıldı | Görüntünün beşte birini kapatıyordu; mesafe ve yön zaten iğne etiketlerinde ve nesne kutularında yazılı |
| Kalibrasyon hedefi YOLO kutusundan | Serbest bağlı bileşen zeminden köprülenip tüm sahneyi "tek cisim" yapıyordu |
| Etiket oylamayla + üstünlük şartı | YOLO kareyi bağımsız işler; yakın skorlu iki sınıf arasında etiket zıplıyordu |
| İzleyici sürüm sayacıyla beslenir | Ana döngü ağdan hızlı; aynı sonucu tekrar oylamak oyları şişirirdi |
| Mesafede ölü bantlı EMA | Duran cisim kare-kare oynuyordu; yaklaşma hızlı alfayla geçtiği için güvenlik bozulmaz |
| `input_size` 518 | 210'da ayrım gücü 1.43x; cisim ve arka duvar aynı ham değeri üretip kalibrasyonu imkânsız kılıyordu |
| Renk LUT'u | Nokta başına `applyColorMap` pencere modunu 7 FPS'e düşürüyordu |
| Asimetrik histerezis | Yaklaşma anında kabul (güvenlik), uzaklaşma teyitli kabul |
