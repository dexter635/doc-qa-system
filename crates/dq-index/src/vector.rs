//! Yogun (dense) vektor indeksi.
//!
//! Bilincli olarak *tam* (exact) tarama secildi. Hedef kurulumdaki korpus
//! buyuklugunde (10^4-10^5 chunk) 384 boyutlu bir taramanin maliyeti
//! milisaniyeler mertebesindedir ve ANN'in (HNSW) getirdigi geri cagirma
//! kaybi, indeks bakim karmasikligi ve ek bagimlilik gerekcelendirilemez.
//! Buyume halinde [`VectorIndex`] arayuzunun arkasina ANN eklenebilir.

use rayon::prelude::*;

pub trait VectorIndex: Send + Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// `allow` ile izin verilen satirlar arasindan en yakin `k` sonucu dondurur.
    /// Rayon paralel taramasi gerektirdigi icin yordam `Sync` olmalidir.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        allow: &(dyn Fn(usize) -> bool + Sync),
    ) -> Vec<(usize, f32)>;
}

/// Satir-oncelikli tek bir `Vec<f32>` icinde tutulan duz indeks.
/// Bellek yerelligi sayesinde cok hizli taranir.
#[derive(Default)]
pub struct FlatIndex {
    dim: usize,
    data: Vec<f32>,
    rows: usize,
}

impl FlatIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
            rows: 0,
        }
    }

    pub fn with_capacity(dim: usize, rows: usize) -> Self {
        Self {
            dim,
            data: Vec::with_capacity(dim * rows),
            rows: 0,
        }
    }

    /// Vektorun birim uzunlukta oldugu varsayilir (bkz. `embed::normalize`).
    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim, "vektor boyutu indeks ile uyusmuyor");
        if v.len() != self.dim {
            // Bozuk veri indeksi kaydirmamali; sifirla doldur.
            let mut fixed = vec![0f32; self.dim];
            let n = v.len().min(self.dim);
            fixed[..n].copy_from_slice(&v[..n]);
            self.data.extend_from_slice(&fixed);
        } else {
            self.data.extend_from_slice(v);
        }
        self.rows += 1;
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.rows = 0;
    }
}

impl VectorIndex for FlatIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        allow: &(dyn Fn(usize) -> bool + Sync),
    ) -> Vec<(usize, f32)> {
        if self.rows == 0 || k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = (0..self.rows)
            .into_par_iter()
            .filter(|i| allow(*i))
            .map(|i| {
                let row = &self.data[i * self.dim..(i + 1) * self.dim];
                let mut acc = 0f32;
                for (a, b) in row.iter().zip(query) {
                    acc += a * b;
                }
                (i, acc)
            })
            .collect();

        let k = k.min(scored.len());
        let pivot = k.saturating_sub(1).min(scored.len().saturating_sub(1));
        scored.select_nth_unstable_by(pivot, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_nearest() {
        let mut idx = FlatIndex::new(3);
        idx.push(&[1.0, 0.0, 0.0]);
        idx.push(&[0.0, 1.0, 0.0]);
        idx.push(&[0.9, 0.1, 0.0]);
        let hits = idx.search(&[1.0, 0.0, 0.0], 2, &|_| true);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 2);
    }

    #[test]
    fn respects_filter() {
        let mut idx = FlatIndex::new(2);
        idx.push(&[1.0, 0.0]);
        idx.push(&[0.0, 1.0]);
        let hits = idx.search(&[1.0, 0.0], 5, &|i| i != 0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }
}
