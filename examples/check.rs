use narashi::Narashi;

fn main() -> anyhow::Result<()> {
    let n = Narashi::new()?;
    let texts: Vec<String> = ["白い背景", "白背景", "漫画", "マンガ", "頬紅", "照れ"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    for threshold in [90.0, 95.0, 97.0] {
        println!("=== threshold: {threshold} ===");
        let groups = n.normalize(&texts, threshold)?;
        for g in &groups {
            println!("  canonical={} members={:?}", g.canonical, g.members);
        }
    }
    Ok(())
}
