## 4.6 main.rs — CLI Entry

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "vitriol-calibrate")]
enum Cli {
    Calibrate {
        #[arg(long)]
        quick: bool,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "calibrated")]
        profile: String,
        #[arg(long)]
        aggressive: bool,
    },
}

fn main() -> Result