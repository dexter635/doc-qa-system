use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Belge gizlilik derecesi (MSY 317-2 / NATO esdegerleri ile hizali).
///
/// Sayisal siralama onemlidir: kullanicinin yetki seviyesi, belgenin
/// derecesinden kucukse erisim reddedilir (Bell-LaPadula "no read up").
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    /// Tasnif Disi / Unclassified
    #[default]
    Unclassified = 0,
    /// Hizmete Ozel / Restricted
    Restricted = 1,
    /// Ozel / Confidential
    Confidential = 2,
    /// Gizli / Secret
    Secret = 3,
    /// Cok Gizli / Top Secret
    TopSecret = 4,
}

impl Classification {
    pub fn label_tr(&self) -> &'static str {
        match self {
            Classification::Unclassified => "TASNIF DISI",
            Classification::Restricted => "HIZMETE OZEL",
            Classification::Confidential => "OZEL",
            Classification::Secret => "GIZLI",
            Classification::TopSecret => "COK GIZLI",
        }
    }

    pub fn all() -> [Classification; 5] {
        [
            Classification::Unclassified,
            Classification::Restricted,
            Classification::Confidential,
            Classification::Secret,
            Classification::TopSecret,
        ]
    }

    /// Bu dereceye sahip bir belgeye `clearance` yetkili mi?
    pub fn readable_by(&self, clearance: Classification) -> bool {
        clearance >= *self
    }

    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => Classification::Restricted,
            2 => Classification::Confidential,
            3 => Classification::Secret,
            4 => Classification::TopSecret,
            _ => Classification::Unclassified,
        }
    }
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label_tr())
    }
}

/// Belge / sorgu dili.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    Tr,
    En,
    #[default]
    Unknown,
}

impl Lang {
    pub fn code(&self) -> &'static str {
        match self {
            Lang::Tr => "tr",
            Lang::En => "en",
            Lang::Unknown => "und",
        }
    }

    pub fn from_code(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "tr" | "tur" | "tr-tr" => Lang::Tr,
            "en" | "eng" | "en-us" | "en-gb" => Lang::En,
            _ => Lang::Unknown,
        }
    }
}

/// Sayfa metninin nasil elde edildigi. Guvenilirlik skorlamasinda kullanilir.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMethod {
    /// PDF gomulu metin katmani (en guvenilir)
    PdfText,
    /// Raster goruntuden OCR
    Ocr,
    /// Once PDF metni denendi, yetersizdi; OCR ile tamamlandi
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentStatus {
    Pending,
    Processing,
    Ready,
    Failed,
}

impl DocumentStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentStatus::Pending => "pending",
            DocumentStatus::Processing => "processing",
            DocumentStatus::Ready => "ready",
            DocumentStatus::Failed => "failed",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "processing" => DocumentStatus::Processing,
            "ready" => DocumentStatus::Ready,
            "failed" => DocumentStatus::Failed,
            _ => DocumentStatus::Pending,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub filename: String,
    pub mime: String,
    /// Icerik hash'i (blake3) - ayni belgenin tekrar islenmesini engeller.
    pub content_hash: String,
    pub size_bytes: u64,
    pub page_count: usize,
    pub lang: Lang,
    pub classification: Classification,
    pub status: DocumentStatus,
    pub owner: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Isleme hatasi varsa kisa aciklama.
    pub error: Option<String>,
    /// Sayfa bazli ortalama OCR guveni (0..1). Metin katmani varsa 1.0.
    pub avg_confidence: f32,
}

/// Isleme hattinin ara ciktisi: tek bir sayfanin cikarilmis metni.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageText {
    pub page_no: usize,
    pub text: String,
    pub method: ExtractionMethod,
    /// 0..1 arasi ortalama karakter/kelime guveni.
    pub confidence: f32,
    pub lang: Lang,
}

/// Vektor indeksine giren en kucuk birim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub doc_id: Uuid,
    /// Belge icindeki sirasi (0'dan baslar) - komsu chunk genisletme icin.
    pub ordinal: usize,
    pub page_from: usize,
    pub page_to: usize,
    pub text: String,
    /// Basliklardan olusan hiyerarsik yol: "3. Bakim > 3.2 Periyodik Kontrol"
    pub heading_path: Option<String>,
    pub token_estimate: usize,
    pub lang: Lang,
    pub classification: Classification,
    pub confidence: f32,
    /// Ust chunk kimligi (parent-child retrieval icin). None = ana chunk.
    pub parent_id: Option<Uuid>,
    /// Chunk turu: "parent" | "child" | "standalone"
    pub chunk_type: ChunkType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChunkType {
    Parent,
    Child,
    Standalone,
}

impl Default for ChunkType {
    fn default() -> Self {
        ChunkType::Standalone
    }
}

/// Arama sonucu: chunk + skorlar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk: Chunk,
    /// Nihai (fusion + rerank sonrasi) skor.
    pub score: f32,
    /// Kosinus benzerligi (0..1) - yoksa None.
    pub dense_score: Option<f32>,
    /// BM25 skoru - yoksa None.
    pub sparse_score: Option<f32>,
    /// Cross-encoder yeniden siralama skoru - yoksa None.
    pub rerank_score: Option<f32>,
    pub doc_filename: String,
}

/// Cevabin dayandigi kaynak referansi.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Cevap metnindeki [1], [2] numarasi.
    pub marker: usize,
    pub doc_id: Uuid,
    pub doc_filename: String,
    pub chunk_id: Uuid,
    pub page_from: usize,
    pub page_to: usize,
    pub snippet: String,
    pub score: f32,
}

/// Cevabin neden guvenilir (ya da guvenilmez) oldugunu anlatan olcum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Groundedness {
    /// Cevaptaki cumlelerin kaynaklarla desteklenme orani (0..1).
    pub support_ratio: f32,
    /// Desteklenmeyen cumleler (denetim ve kullanici uyarisi icin).
    pub unsupported_sentences: Vec<String>,
    /// En iyi kaynagin retrieval skoru.
    pub top_score: f32,
    /// Esik degerlerini gecti mi?
    pub passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerKind {
    /// Kaynaklarla desteklenen normal cevap.
    Grounded,
    /// Yeterli kaynak bulunamadi -> bilerek cevap verilmedi.
    Refused,
    /// Guardrail tarafindan bloklandi.
    Blocked,
}

/// Ajan dongusunun tek bir adiminin turu (kullaniciya/denetime seffaflik icin).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStepKind {
    /// Sorgu ayristirma ve arac (belge kapsami) secimi.
    Plan,
    /// Bir alt-sorgu icin hibrit arama calistirildi.
    Retrieve,
    /// LLM (veya cikarimsal yedek) ile cevap uretildi.
    Generate,
    /// Cikti guardrail'i cevabi degerlendirdi; yetersizse yeniden deneme tetiklendi.
    Critique,
}

/// Ajanin attigi tek bir adimin kaydi. `Answer.trace` icinde tasinir; kullanici
/// arayuzunde "ajan adimlari" olarak, denetimde ise karar gerekcesi olarak kullanilir.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub step: usize,
    pub kind: AgentStepKind,
    /// Insan-okunur kisa aciklama ("2 alt sorguya ayristirildi" gibi).
    pub description: String,
    /// Makine tarafindan islenebilir ayrinti (alt sorgular, skorlar, sureler).
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub query_id: Uuid,
    pub kind: AnswerKind,
    pub text: String,
    pub citations: Vec<Citation>,
    pub groundedness: Groundedness,
    pub lang: Lang,
    /// Cevabin en yuksek dereceli kaynagindan miras alinan gizlilik derecesi.
    pub classification: Classification,
    pub cached: bool,
    pub latency_ms: u64,
    pub model: String,
    /// Guardrail / pipeline uyarilari (kullaniciya gosterilir).
    pub warnings: Vec<String>,
    /// Ajanin izledigi adimlar (planlama, arama, uretim, elestiri/yeniden deneme).
    #[serde(default)]
    pub trace: Vec<AgentStep>,
}

/// Kullanici oturum baglami; yetkilendirme ve denetim kaydi icin tasinir.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub username: String,
    pub clearance: Classification,
    pub roles: Vec<String>,
}

impl UserContext {
    pub fn is_admin(&self) -> bool {
        self.roles.iter().any(|r| r == "admin")
    }
}

/// Degistirilemez denetim kaydi (audit log) satiri.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: Uuid,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub action: String,
    pub subject: Option<String>,
    pub outcome: String,
    pub detail: serde_json::Value,
    /// Zincirleme hash: onceki kaydin hash'i + bu kayit -> kurcalama tespiti.
    pub prev_hash: String,
    pub hash: String,
}
