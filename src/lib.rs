use anyhow::{Result, anyhow};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::collections::HashMap;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "intfloat/multilingual-e5-small";

#[derive(Debug, Clone)]
pub struct Group {
    pub canonical: String,
    pub members: Vec<String>,
}

pub struct Narashi {
    embedder: TextEmbedding,
    tokenizer: Tokenizer,
}

impl Narashi {
    pub fn new() -> Result<Self> {
        let embedder = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small),
        )?;
        let tokenizer = Tokenizer::from_pretrained(MODEL_REPO, None)
            .map_err(|e| anyhow!("tokenizer load failed: {e}"))?;
        Ok(Self { embedder, tokenizer })
    }

    pub fn similarity(&self, a: &str, b: &str) -> Result<f32> {
        let inputs = vec![format!("query: {a}"), format!("query: {b}")];
        let embeddings = self.embedder.embed(inputs, None)?;
        Ok(cosine_to_score(cosine_similarity(
            &embeddings[0],
            &embeddings[1],
        )))
    }

    pub fn normalize(&self, texts: &[String], threshold: f32) -> Result<Vec<Group>> {
        let n = texts.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let inputs: Vec<String> = texts.iter().map(|t| format!("query: {t}")).collect();
        let embeddings = self.embedder.embed(inputs, None)?;

        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_to_score(cosine_similarity(&embeddings[i], &embeddings[j]));
                if sim >= threshold {
                    uf.union(i, j);
                }
            }
        }

        let keys: Vec<(usize, u64)> = texts
            .iter()
            .map(|t| self.generality_key(t))
            .collect::<Result<Vec<_>>>()?;

        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            buckets.entry(uf.find(i)).or_default().push(i);
        }

        let mut groups: Vec<Group> = buckets
            .into_values()
            .map(|indices| {
                let canonical_idx = *indices.iter().min_by_key(|&&i| keys[i]).unwrap();
                let mut members: Vec<String> =
                    indices.iter().map(|&i| texts[i].clone()).collect();
                members.sort();
                Group {
                    canonical: texts[canonical_idx].clone(),
                    members,
                }
            })
            .collect();

        groups.sort_by(|a, b| a.canonical.cmp(&b.canonical));
        Ok(groups)
    }

    fn generality_key(&self, text: &str) -> Result<(usize, u64)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize failed: {e}"))?;
        let ids = encoding.get_ids();
        let count = ids.len();
        let sum: u64 = ids.iter().map(|&id| id as u64).sum();
        Ok((count, sum))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn cosine_to_score(cos: f32) -> f32 {
    ((cos + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0)
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.find(p);
        }
        self.parent[x]
    }
    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent[rx] = ry;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let a = [1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn score_range() {
        assert!((cosine_to_score(1.0) - 100.0).abs() < 1e-4);
        assert!((cosine_to_score(0.0) - 50.0).abs() < 1e-4);
        assert!((cosine_to_score(-1.0) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn union_find_merges() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(1, 2);
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(3));
    }

    #[test]
    #[ignore]
    fn real_similarity_high() {
        let n = Narashi::new().unwrap();
        let s = n.similarity("猫", "ネコ").unwrap();
        assert!(s > 70.0, "expected high similarity, got {s}");
    }

    #[test]
    #[ignore]
    fn real_normalize_groups() {
        let n = Narashi::new().unwrap();
        let texts: Vec<String> = ["猫", "ネコ", "犬", "イヌ", "自動車"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let groups = n.normalize(&texts, 80.0).unwrap();
        for g in &groups {
            println!("canonical={} members={:?}", g.canonical, g.members);
        }
        assert!(!groups.is_empty());
    }
}
