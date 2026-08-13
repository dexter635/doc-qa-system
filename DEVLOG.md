# DEVLOG — Geliştirme Süreci Kaydı

Bu dosya bir final rapor değil, geliştirme sırasında alınan kararların ve
karşılaşılan sorunların kronolojik kaydıdır.

## 1. Problemi nasıl parçaladık?

Case study, dört zorunlu işlev tanımlıyordu: belge yükleme (PDF/JPG/PNG),
metin çıkarımı (TR/EN, taranmış belgeler dahil), soru-cevap, halüsinasyon
kontrolü, ve bir arayüz. Bunu tek bir uygulama yerine **bağımsız test
edilebilir katmanlara** ayırdık, çünkü:

1. "Halüsinasyon üretmemeli" gereksinimi tek başına ciddi bir mühendislik
   problemi — bunu ayrı bir `dq-guard` katmanına çıkarıp LLM'e "güvenmeyen"
   bağımsız bir doğrulama mekanizması olarak tasarladık.
2. Savunma sanayii bağlamı örtük olarak şunu gerektiriyordu: yerel model
   zorunluluğu, gizlilik derecesine göre erişim kontrolü, değiştirilemez
   denetim kaydı. Bunlar dokümanda açıkça yazmasa da "sistemle ilgili
   belirtilmeyen teknik kararlar için inisiyatif alın" notuyla bekleniyordu.

Sıralama: `dq-core` (ortak tipler/config/hata) → `dq-ingest` (belge işleme)
→ `dq-index` (arama+depolama) → `dq-llm` → `dq-guard` → `dq-rag`
(orkestrasyon) → `dq-server` (API) → `dq-web` (arayüz). Her katman bir
öncekini derleyip test ettikten sonra yazıldı; böylece bir alt katmandaki bir
hata üst katmanlara taşınmadan yakalandı.

## 2. Ortam kurulumu — ilk engel

Makinede ne Rust ne de bir C++ derleyici kuruluydu. `rustup` ile Rust
kurulumu sorunsuzdu, ancak ilk `cargo build` **`link.exe` bulunamadı**
hatasıyla durdu: Windows'ta Rust'ın varsayılan MSVC hedefi, Visual Studio
Build Tools + Windows SDK gerektiriyor (VS Code'un kendisi yeterli değil).

- `winget install Microsoft.VisualStudio.2022.BuildTools` ile kurulum
  başlatıldı, ancak bu kurulum **yönetici (UAC) onayı** istiyordu ve
  otomatik bir ajan bunu geçemez. Kullanıcıdan bu adımı tamamlamasını
  istedik; o sırada Rust'a bağımlı olmayan dosyaları (ingest/pdf.rs,
  imageproc.rs, ocr.rs, chunk.rs gibi saf mantık dosyaları) yazmaya devam
  ettik ki zaman kaybı olmasın.
- Kurulum tamamlandıktan sonra bile `Windows Kits\10\Lib` dizini yoktu
  (Build Tools C++ araçlarını kurmuş ama Windows SDK bileşenini
  kurmamıştı) — bu ikinci bir "sessiz" engeldi, sadece derleme hatasından
  değil dizin varlığını kontrol ederek fark edildi. Kullanıcı SDK'yı da
  tamamladıktan sonra derleme sorunsuz ilerledi.

**Ders**: Windows'ta Rust için MSVC toolchain kurulumunu asla varsayma;
`link.exe` VE `kernel32.lib`'in ikisinin de var olduğunu doğrula.

## 3. Denenen ve vazgeçilen yaklaşımlar

- **HNSW (yaklaşık en yakın komşu) indeksi**: `hnsw_rs` crate'i
  değerlendirildi. Hedef korpus büyüklüğü (belge başına birkaç yüz chunk,
  toplamda 10⁴–10⁵ mertebesi) için 384 boyutlu tam (exact) kosinüs
  taramasının maliyeti milisaniyeler mertebesinde kalıyor; ANN'in getirdiği
  geri çağırma (recall) kaybı ve ek bağımlılık karmaşıklığı bu ölçekte
  gerekçelendirilemedi. Bunun yerine `VectorIndex` trait'i arkasında düz
  (flat) bir indeks yazıldı; büyüme durumunda trait'in arkasına HNSW
  eklenebilir.
- **pdfium/mupdf ile PDF render**: Taranmış PDF'lerin sayfalarını rastere
  çevirmek için önce harici bir PDF render kütüphanesi düşünüldü, ancak bu
  ek bir native bağımlılık (ve Windows'ta ayrı bir derleme sorunu riski)
  demekti. Bunun yerine `lopdf` ile PDF içindeki **gömülü görüntü
  nesneleri** doğrudan çıkarılıp OCR'a veriliyor — taranmış belgelerin
  standart yapısında (sayfa başına tek büyük görüntü) bu yeterli ve daha az
  bağımlılık gerektiriyor.
- **Sınıflandırıcı model ile prompt injection tespiti**: Daha kapsamlı
  olurdu, ama kapalı ağda ek bir model = ek gecikme + ek belirsizlik +
  açıklanamayan kararlar. Kural tabanlı (regex) bir yaklaşıma geçildi; her
  kural adı ve ağırlığıyla denetim kaydına yazılıyor.
- **leptos_router ile çok sayfalı yönlendirme**: Yalnızca 4 ekran (giriş,
  belgeler, soru-cevap, denetim) olduğu için tam bir router yerine tek bir
  `RwSignal<Tab>` ile sekme yönetimi tercih edildi — daha az bağımlılık, daha
  az versiyon uyuşmazlığı riski.

## 4. Kritik karar noktaları

- **Groundedness'i nasıl ölçeceğiz?** İki seçenek vardı: (a) ikinci bir LLM
  çağrısıyla "bu cevap kaynaklarla destekleniyor mu?" diye sormak, (b)
  modelsiz, deterministik bir metin örtüşme ölçüsü. (a) ek gecikme + ek
  belirsizlik getiriyordu ve *LLM'in kendi hatasını LLM'e doğrulatmak*
  döngüsel bir güven zinciri oluşturuyordu. (b)'yi seçtik: cevaptaki her
  cümle, atıf verdiği kaynak parçayla karakter n-gram *containment* oranı
  üzerinden karşılaştırılıyor. Modelsiz, hızlı, ve tersine mühendisliği
  (neden reddedildiği) açıklanabilir.
- **Cevap dili nasıl garanti edilir?** Prompt'ta talimat vermek yeterli
  değil (modeller bazen talimatı yok sayar). Çıktı guardrail'i ayrıca
  `detect_lang` ile cevabın dilini soru diliyle karşılaştırıp uyumsuzlukta
  uyarı ekliyor (sert reddetmek yerine, çünkü teknik terimler/sayılar iki
  dilde de ortak olabilir ve yanlış pozitif riski yüksek).
- **Belge içeriğindeki "talimat" enjeksiyonu**: Bir PDF'in içine
  "önceki talimatları yoksay" gibi bir cümle gömülmüşse, bu LLM'e
  *veri* olarak gitmeli, *komut* olarak değil. Bağlam bloğu
  `<belgeler>...</belgeler>` etiketleriyle sarmalanıyor ve belge içeriğinde
  bu etiketlere benzeyen dizeler (`</belgeler>`, `<|im_start|>` vb.)
  zararsızlaştırılıyor (bkz. `dq-llm/src/prompts.rs::sanitize_context`).

## 5. Nerede takıldık, nasıl çözdük?

1. **`tr_lower` bug'ı (en öğretici olan)**: Girdi guardrail testlerinden
   biri ("Ignore all previous instructions...") beklenmedik şekilde
   *geçiyordu* (engellenmiyordu). Kök neden: Türkçe'ye duyarlı küçük harfe
   çevirme fonksiyonumuz ASCII `'I'` harfini Türkçe kuralına göre noktasız
   `'ı'`ya çeviriyordu — yani "Ignore" önce "ıgnore" oluyordu ve İngilizce
   regex deseni ("ignore") artık eşleşmiyordu. Çözüm: `tr_lower`'ı olduğu
   gibi bırakıp (arama/sıralama için hâlâ doğru), guardrail eşleştirmesi
   için ayrı bir `casefold()` fonksiyonu eklendi (yalnızca 'İ' özel
   durumunu ele alır, ASCII 'I' standart 'i' olur). Bu, karma TR/EN metin
   işleyen her regex tabanlı koda uygulanması gereken genel bir ders.
2. **Leptos'ta `FnOnce` hatası**: Belge tablosunda her satır için "Sil"
   butonu oluştururken, dışarıda tanımlı **tek bir** silme closure'ını
   (`String` yakalayan, dolayısıyla `Copy` olmayan) her satırda yeniden
   "taşımaya" (move) çalışmak, listeyi oluşturan `.map()` closure'ının
   yalnızca bir kez çağrılabilir (`FnOnce`) olmasına yol açtı — ki
   reaktif render için `FnMut`/`Fn` gerekiyordu. Çözüm: paylaşılan
   closure'ı kaldırıp her satırda ihtiyaç duyulan `token`/`id` değerlerini
   taze `.clone()`layıp tıklama işleyicisini satır içi (inline)
   oluşturmak.
3. **`rayon` ile `dyn Fn` sınırı**: Paralel filtrelemede kullanılan
   `&dyn Fn(usize) -> bool` parametresi `Sync` değildi; `rayon`'un paralel
   yineleyicileri `Sync` gerektiriyor. `&(dyn Fn(usize) -> bool + Sync)`
   olarak düzeltildi.
4. **`fastembed`/ONNX çağrılarında `&mut self` gereksinimi**: `Mutex`
   guard'ı değişmez (`let guard`) olarak tanımlanmıştı; `TextEmbedding::embed`
   ve `TextRerank::rerank` içeride `&mut self` istiyor. `let mut guard`
   ile düzeltildi.
5. **Bazı testler yanlış yazılmıştı (kod değil)**: `context_respects_token_budget`
   testi, tek bir chunk'ın bile bütçeyi aşacağı kadar büyük veri
   kullanıyordu — oysa kodun kuralı "en az bir chunk her zaman dahil
   edilir" (cevabın tamamen bağlamsız kalmaması için). Test verisi küçültülüp
   ayrıca bu istisnayı doğrulayan yeni bir test eklendi.

## 6. Zamanı nasıl harcadık?

Kabaca: %15 ortam kurulumu (Rust/MSVC/SDK/Trunk, çoğu bekleme süresi),
%35 çekirdek+ingest+index (PDF/OCR/chunking/hibrit arama en çok
mühendislik gerektiren kısım), %20 guardrail (injection kuralları + PII
regex'leri + groundedness), %15 server (auth/audit/rota), %15 frontend
(Leptos+Trunk, ilk kurulumu ve tek closure hatası dışında hızlı ilerledi).

## 7. Baştan başlasak neyi farklı yapardık?

- Cross-encoder reranker'ı (şu an opsiyonel/kapalı) varsayılan olarak
  değerlendirip gerçek bir Türkçe teknik doküman setinde ölçüm yapardık;
  şu an yalnızca RRF+MMR ile sınırlı test verisiyle doğrulandı.
- `dq-core::text::casefold` gibi ASCII-güvenli yardımcıları en baştan,
  Türkçe metin fonksiyonlarıyla birlikte tasarlardık (sonradan eklemek
  yerine) — bu, guardrail bug'ının baştan önüne geçerdi.
- Gerçek bir Tesseract kurulumuyla taranmış Türkçe teknik doküman üzerinde
  uçtan uca OCR kalitesi ölçümü zaman kısıtı nedeniyle yapılamadı; bu
  [TESTING.md](TESTING.md)'de açıkça bir sınırlama olarak belirtildi.

## 8. Agentic RAG döngüsü eklentisi

İlk sürüm tek adımlı (klasik) RAG olarak teslim edilmişti: tek retrieve →
tek generate → output guardrail. Soru-cevap kalitesini artırmak ve "belgede
olmayan bilgi üretmeme" gereksinimini daha güçlü karşılamak için, LLM-koordineli
çok adımlı bir ajana genişlettik:

- **Plan**: LLM sorguyu alt-sorgulara böler ve (belirtilmişse) hangi belgenin
  taranması gerektiğini önerir. LLM yoksa veya JSON ayrıştırılamazsa sezgisel
  yedeğe (orijinal sorgu, filtresiz) düşer — bu adım sistemi asla durdurmaz.
- **Retrieve**: Her alt-sorgu için hibrit arama çalışır ve sonuçlar chunk
  kimliğine göre birleştirilir; en yüksek per-sorgu skoru korunur.
- **Generate**: Bağlamdan LLM (veya extractive yedek) cevap üretir.
- **Critique**: Output guardrail'i source-grounded doğrular. Yetersizse →
  sorgu yeniden formüle edilip döngü tekrarlanır.

**Karar gerekçesi**: Savunma sanayii, serbest dolaşan çok adımlı ajanı
kabul etmez. Bu yüzden:
- `agent.max_steps` sabit bir tavan (varsayılan 2), maliyet/gecikmeyi
  öngörülebilir tutar.
- `enable_query_decomposition`, `enable_self_correction`,
  `enable_tool_doc_selection` bayraklarıyla her davranış kapatılabilir;
  kapalıyken sistem klasik tek-adımlı RAG'e döner.
- Her adım `Answer.trace` içinde (`Plan/Retrieve/Generate/Critique` türleriyle)
  toplanır; kullanıcı arayüzünde "ajan adımları" olarak, denetim kaydında
  karar gerekçesi olarak kullanılır.

Bu katman `dq-rag/src/agent.rs` içindedir ve iki entegrasyon testi içerir
(LLM yokken extractive yedeğe zarif düşüş; ilk cevap grounded değilse
self-correction yeniden denemesi). JSON çıktısı `dq-llm::json` ile çıkarılır
(bkz. bölüm 9).

## 9. Yerel modellerde yapısal JSON çıktısı: `dq-llm::json`

Küçük parametreli yerel modeller (7B mertebesi) "structured output" /
function-calling'i her zaman düzgün desteklemez; çıktıyı kod bloğuna sarabilir,
önüne/sonuna açıklama ekleyebilir. Bu yüzden planlama/self-correction
çağrılarından gelen metinden, ilk dengeli `{...}` bloğunu çekip ayıklayan bir
yardımcı yazıldı (`extract_json_object`). Ayrıştırma başarısız olursa çağıran
taraf sezgisel yedeğe döner; model çıktısı hiçbir yerde panic'e ya da
istisnaya yol açmaz. Bu katman 5 birim testi ile doğrulanır.

## 10. Üretim/CI derleme — ONNX Runtime linker hatası

CI'daki Docker build, backend (`dq-server`) bağlama aşamasında duruyordu:

```
rust-lld: undefined symbol: __isoc23_strtoll / __isoc23_strtoull / __isoc23_strtol
rust-lld: undefined symbol: std::__cxx11::basic_string<...>::_M_replace_cold
```

Kök neden: `fastembed`/`ort` tarafından indirilen **önceden derlenmiş
ONNX Runtime** ikilisi, daha yeni glibc (2.38+) ve GCC 13 libstdc++ sembolleri
(`__isoc23_strto*`, `_M_replace_cold`) kullanıyor. Ancak build imajları
**Debian 12 bookworm** tabanlıydı (glibc 2.36 + GCC 12); istendiği için bu
semboller sistemde yoktu ve bağlama başarısız oluyordu.

Çözüm: tüm imajları **Debian 13 (trixie)**'ye yükseltmek
(`rust:1-slim-trixie` ve `debian:trixie-slim`); trixie glibc 2.41 + GCC 14
içerir. Ayrıca trixie'de `libstdc++-12-dev` paketi bulunmadığından
`build-essential` kapsamındaki `libstdc++-14-dev`'e geçildi. Runtime imajı da
aynı glibc sürümünü paylaşması için trixie'ye çekildi — derleyici ve çalıştırıcı
aynı libc üzerinde kaldığından "önce derle, sonra çatla" sınıfındaki gizli uyum
sorunları baştan engellenir.

**Ders**: Rust'ın C/C++ araç zinciri kadar, hazır (prebuilt) native ikililerin
(ort/fastembed gibi) **onların derlendiği glibc/libstdc++'ını** da hesaba
katmak gerekir; "slim" (en güncel olmayan) aylık-etiketli Debian imajları bu
bağlamda sessiz bir tuzak olabilir.
