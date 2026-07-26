# EdgeVision — Teknik Dokümantasyon

Tek kameralı, metrik derinlik tabanlı mesafe kestirimi ve engel kaçınma sistemi.

---

## 1. Sistem Nedir, Ne Yapar

**Amaç:** Sıradan bir web kamerasıyla, ek hiçbir donanım (derinlik sensörü, stereo kamera, LiDAR, ToF) kullanmadan, görüş alanındaki her yön için **metre cinsinden mesafe** üretmek ve bunu engel kaçınma komutlarına çevirmek.

**Donanım kısıtı:** Kesinlikle tek kamera. Sistem başka hiçbir sensöre bağlı değildir.

**Temel yaklaşım:** Mesafe tek bir kaynaktan gelir — Depth Anything V2 (metrik, iç mekan) sinir ağı. Bu ağ her piksel için doğrudan metre değeri üretir. Klasik monoküler geometri yığını (zemin düzlemi varsayımları, nesne boyutu öğrenme, ArUco kalibrasyonu, MiDaS füzyonu) tamamen kaldırılmıştır çünkü bu model onları gereksiz kılar.

### Ölçülen sonuçlar

| Ana döngü hızı | **30.2–30.9 FPS** |
| İşlem + çizim maliyeti | 4.88 ms/kare (33 ms) |
| Derinlik ağı tazeleme | 25–31 FPS (29–40 ms/çıkarım, DirectML) |
| Kalibre mesafe aralığı | 0.54 – 6.57 m (3 noktalı kalibrasyonla) |
| Zemin düzlemi doğruluğu |

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
   ├─► GroundRemover: zemin düzlemi uydur, zemin maskesi üret
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
   └─► viz.compose_view(): kamera + radar | derinlik haritası
```

### İki iş parçacığı neden şart

Derinlik ağı ViT tabanlıdır ve tek başına ~25-30 FPS koşar. Eğer ana döngüyü bloke etseydi tüm sistem o hıza düşerdi. Ağ ayrı bir iş parçacığında sürekli koşar ve en son haritayı paylaşımlı olarak yayınlar; ana döngü bu haritayı bloke olmadan okur. Böylece ana döngü kamera hızında (30 FPS) akarken derinlik arka planda tazelenir.

---

## 3. Dosya Dosya Referans

### `main.py` — Giriş noktası

Argüman ayrıştırma, kamera döngüsü, pencere yönetimi, klavye/fare olayları.

| Çalışma modu | Komut |
|---|---|
| Pencere (varsayılan) | `python main.py` |
| Video dosyası | `python main.py --source test.mp4` |
| Robot köprüsü (JSON) | `python main.py --json` |
| Penceresiz, konsol durumu | `python main.py --headless` |

**Klavye:** `q` çıkış · `s` kare kaydet · `d` sağ panel değiştir · `c` kalibrasyon modu
**Fare:** Tıklama → o pikselin mesafesi (kalibre + ham değer birlikte)

**Önemli detay:** Kenar haritası yalnızca sağ panelde gösterilecekse hesaplanır (`want_edges`). Gösterilmeyecek bir ürünü hesaplamak boşa CPU yakar.

---

### `edge_vision/config.py` — Yapılandırma

YAML dosyasını dataclass'lara yükler. Bilinmeyen anahtarlar sessizce atılır (eski config dosyaları kırılmaz).

| Sınıf | Görev |

| `CameraConfig` | Kaynak, çözünürlük, FPS, yatay görüş açısı |
| `ProcessingConfig` | İşleme genişliği, ROI üst sınırı, sütun adımı |
| `EdgeConfig` | Canny kenar tespiti (yalnız görselleştirme) |
| `DepthConfig` | Model yolu, girdi boyutu, sağlayıcılar, **kalibrasyon**, renk skalası |
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

**`ObstacleReport`** (dataclass) — Bir karenin tüm çıktısı: engel listesi, en yakın engel, ham tarama dizileri.

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

Renk ↔ mesafe ilişkisinin tek tanımıdır. Derinlik haritası, radar noktaları ve engel işaretleri renklerini buradan alır — böylece bir renk **nerede görünürse görünsün aynı mesafeyi** ifade eder.

**İki parçalı ölçek:**

| Aralık | Renk davranışı |
|---|---|
| `viz_near_m` → `viz_red_max_m` (0.05 → 0.40 m) | **Tamamen kırmızı**, doğrusal kademe (0.05 en koyu) |
| `viz_red_max_m` → `viz_far_m` (0.40 → 3.0 m) | Turuncu → sarı → yeşil → mavi (karekök) |

**Ölçek SABİTTİR, sahneye göre kaymaz.** Kayan bir ölçekte aynı renk her karede başka mesafe demek olur ve sisteme dayanan kararlar sürekli hata üretir.

**Palet detayı (ölçüldü):** TURBO paletinin uçları karanlıktır — `dn=255` koyu bordo (R=122), `dn=0` koyu mor. Bu yüzden en yakın cisim "kırmızı" değil koyu görünüyordu. Ölçek paletin canlı bölgesine (`dn ∈ [22, 228]`) sıkıştırıldı; net kırmızı sınırı `dn ≥ 210`.

**`LUT` (256 girişlik önceden hesaplanmış tablo) — kritik performans düzeltmesi:** `applyColorMap`'i 1×1 dizide çağırmak ~0.83 ms sürüyor. Radar her tarama noktası için bunu çağırınca kare başına **~131 ms** yanıyordu ve pencere modu 30 yerine **~7 FPS**'te koşuyordu. Tablo aramasıyla aynı renk mikrosaniyede bulunur; üretilen renkler birebir aynıdır.

| | Önce | Sonra |
|---|---|---|
| `color_at` | 0.83 ms | **0.019 ms** |
| Radar çizimi | 135 ms | **0.27 ms** |
| `compose_view` | 141 ms | **4.09 ms** |

---

### `edge_vision/viz.py` — Görselleştirme

| Fonksiyon | Görev |
|---|---|
| `compose_view()` | Yan yana iki panel üretir (kamera \| derinlik veya kenar) |
| `draw_debug()` | Sol panel: yön iğneleri + radar |
| `_draw_radar()` | Kuşbakışı radar |
| `_draw_pins()` | Engel yön iğneleri |
| `edge_view()` | Kenar haritası paneli |

**Neden kuşbakışı radar:** Mesafeyi kamera görüntüsünün üstüne çizgi/çubuk olarak bindirmenin her denemesi başarısız oldu (zikzak polyline, dolgulu dikdörtgen, yüzen çubuk). Sebep: 2B kamera görüntüsünde bir çubuğun **dikey konumu mesafeyi doğal olarak temsil etmez**. Sözde-LiDAR verisinin doğru gösterimi kuşbakışı radardır — kamera altta ortada, ileri mesafe (Z) yukarı, yanal konum (X) yana. Her engel gerçek (X, Z) konumuna oturur; mesafe ve yön aynı anda, menzil halkalarıyla okunur. Bu robotikte standart gösterimdir.

Radar tüm çizimi kutunun **alt-tuvaline** yapar; taşan halkalar/noktalar otomatik kırpılır ve kamera görüntüsüne bulaşmaz.

Kamera görüntüsünde yalnızca ince **yön iğneleri** kalır (hangi yatay yönde engel var); mesafe okuması radardan gelir.

---

### `edge_vision/calib_overlay.py` — Pencere-içi kalibrasyon

**`CalibOverlay`** — Terminal gerektirmez; tüm etkileşim OpenCV penceresindedir.

| Tuş | İşlev |
|---|---|
| `c` / `ESC` | Kalibrasyon modunu aç/kapa |
| rakam, `.` | Gerçek mesafeyi yaz |
| `Backspace` | Sil |
| `ENTER` | Ölçümü kaydet (12 kare toplanır) |
| `f` | Uygula + config'e yaz |
| `x` | Noktaları temizle |
| `r` | Başlangıç kalibrasyonuna dön |

**`measure_object()` — cismi ayıklama:** Nişan kutusundaki **en ön yüzey** bağlı bileşen analiziyle (`connectedComponents`) ayrılır ve parlak sarı konturla çizilir; geri kalan her şey bulanıklaştırılıp karartılır. Böylece kullanıcı neyin ölçüldüğünü gözle doğrular.

**Neden düz medyan değil:** Cisim kutuyu doldurmuyorsa düz medyan arka planla karışır. Sentetik testte: eski düz medyan **3.00 m** (arka plan!), yeni yöntem **0.60 m** (doğru).

**Neden noktalar HAM değerden alınır:** Kalibre değerden alınsaydı her kalibrasyon bir öncekinin üstüne biner, noktalar farklı referans çerçevelerine dağılır ve arka arkaya kalibrasyon sistemi bozardı. Ham değer mutlak olduğu için noktalar biriktirilebilir ve `f`'ye tekrar basmak zararsızdır.

**Güvenlik kontrolleri:**

| Kontrol | Amaç |
|---|---|
| Kararsız ölçüm reddi | 12 karenin sapması yüksekse nokta kaydedilmez |
| Monotonluk | Ham artarken gerçek mesafe de artmalı (yoksa model o aralığı ayırt edemiyor) |
| Aynı ham değer | İki farklı mesafede aynı ham → model doygunlukta |
| **Aralık genişliği** | `span < 0.30 m` **ve** `oran < 2.0` ise reddet |
| Kapsama bildirimi | Uygulandığında kapsanan aralık yazılır, 0.5 m altı nokta yoksa uyarı |

**Aralık genişliği kontrolü neden kritik:** Gerçek bir vakada kullanıcı 0.61 m ve 0.70 m'de (yalnızca 9 cm arayla) iki nokta girdi. Bu iki noktanın eğimi (0.168) ekstrapolasyonla **tüm menzili** yönetince sahnenin tamamı 0.49–1.08 m'ye sıkıştı ve derinlik haritasından mavi tamamen kayboldu.

---

## 4. Kalibrasyon Sistemi (Detaylı)

### Ham değer vs kalibre değer

| | Ham | Kalibre |
|---|---|---|
| Nedir | Modelin doğrudan çıktısı | Ham'a dönüşüm uygulanmış hali |
| Uzakta | İsabetli (model eğitim alanında) | Kalibrasyona bağlı |
| Çok yakında (5 cm) | Yanlış (~0.75) | Düzeltilmiş |
| Kim kullanır | Yalnızca kalibrasyon ölçümleri | Ekran, radar, engel kararları |

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

`c` ile kalibrasyona girin ve **birbirinden uzak** mesafeler girin:

| Nokta | Öneri |
|---|---|
| 1 | 0.05 m (el mercekte) |
| 2 | 0.30 m |
| 3 | 1.0 m |
| 4 | 2.0 m |
| 5 | 3–4 m |

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
| `model` | ONNX yolu | Depth Anything V2 metrik model |
| `input_size` | 210 | 14'ün katı olmalı (ViT yaması) |
| `providers` | DirectML, CPU | Denenecek ONNX sağlayıcıları |
| `threads` | 4 | CPU iş parçacığı sayısı |
| `median_ksize` | 3 | Uzamsal median (0=kapalı, 3 veya 5) |
| `calib_scale` / `calib_offset` | 0.514 / −0.147 | Afin kalibrasyon |
| `min_valid_m` | 0.05 | En yakın okuma tabanı |
| `calib_points` | `[]` | Parçalı kalibrasyon tablosu |
| `viz_red_max_m` | 0.40 | Kırmızı bandın üst sınırı |
| `viz_near_m` / `viz_far_m` | 0.05 / 3.0 | Renk skalası uçları |

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

### Girdi çözünürlüğü hakkında

Ölçüldü: `input_size` 210 → 518 modelin yakın/uzak ayrım gücünü **1.64x'ten 3.46x'e** çıkarır ve yalnızca ~10 ms maliyeti vardır (boş kapasite bunu rahat kaldırır).

**Ancak** çözünürlük değişimi ham değer dağılımını tamamen değiştirir ve mevcut kalibrasyonu geçersiz kılar. Denenecekse **önce** parçalı kalibrasyon yapılmalı, **sonra** çözünürlük değiştirilip **yeniden** kalibre edilmelidir.

---



## 8. Kurulum

```bash
pip install -r requirements.txt
```

Windows'ta GPU hızlandırma için ek olarak:
```bash
pip install onnxruntime-directml
```

Model dosyası pip ile gelmez; `models/depth_anything_v2_metric_indoor.onnx` (~95 MB) elle indirilmelidir.

---

## 9. Tasarım Kararlarının Özeti

| Karar | Gerekçe |
|---|---|
| Tek mesafe kaynağı (metrik derinlik ağı) | Klasik monoküler geometri yığını kırılgandı ve kalibrasyona bağımlıydı |
| Ayrı iş parçacığında derinlik | Ana döngünün 30 FPS'te kalması için |
| Zemin düzlemi ayıklama | Zemin, sütun taramasını domine edip gerçek cisimleri gizliyordu |
| Parçalı kalibrasyon | Tek doğru yakın ve uzağı aynı anda tutturamaz |
| Kalibrasyon noktaları hamdan | Kalibre değerden alınırsa üst üste kalibrasyonlar birbirini bozar |
| Sabit renk ölçeği | Kayan ölçekte aynı renk her karede başka mesafe demek olur |
| Kuşbakışı radar | 2B görüntüde dikey konum mesafeyi temsil etmez |
| Renk LUT'u | Nokta başına `applyColorMap` pencere modunu 7 FPS'e düşürüyordu |
| Asimetrik histerezis | Yaklaşma anında kabul (güvenlik), uzaklaşma teyitli kabul |
