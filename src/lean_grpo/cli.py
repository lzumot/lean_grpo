"""Command-line interface for Lean GRPO training."""

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lean_grpo.inference_client import InferenceClient, MockInferenceClient, VLLMClient
from lean_grpo.lean_interface import LeanInterface, MockLeanInterface
from lean_grpo.reward import (
    REWARD_CONFIG_BINARY,
    REWARD_CONFIG_LENIENT,
    REWARD_CONFIG_SHAPED,
    REWARD_CONFIG_STRICT,
    RewardConfig,
)
from lean_grpo.trainer import LeanGRPOConfig, LeanGRPOTrainer, LeanGRPOPipeline

app = typer.Typer(help="Lean GRPO: Train LLMs to generate Lean 4 proofs")
console = Console()


REWARD_CONFIGS = {
    "binary": REWARD_CONFIG_BINARY,
    "shaped": REWARD_CONFIG_SHAPED,
    "lenient": REWARD_CONFIG_LENIENT,
    "strict": REWARD_CONFIG_STRICT,
}

ALGORITHMS = ["grpo", "dgpo", "drgrpo", "dapo", "gspo"]


@app.command()
def train(
    data_path: Path = typer.Argument(
        ..., help="Path to JSON/JSONL file with theorems"
    ),
    output_dir: Path = typer.Option(
        "outputs", "--output", "-o", help="Output directory"
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        "--model", "-m",
        help="Base model to fine-tune",
    ),
    inference_url: str = typer.Option(
        "http://localhost:8000/v1",
        "--inference-url",
        help="URL for inference API",
    ),
    inference_key: str = typer.Option(
        None,
        "--inference-key",
        help="API key for inference (defaults to env var)",
    ),
    algorithm: str = typer.Option(
        "grpo",
        "--algorithm", "-a",
        help=f"RL algorithm: {', '.join(ALGORITHMS)}",
    ),
    num_generations: int = typer.Option(
        8, "--num-generations", "-n", help="GRPO group size"
    ),
    learning_rate: float = typer.Option(
        5e-6, "--lr", help="Learning rate"
    ),
    num_epochs: int = typer.Option(
        1, "--epochs", "-e", help="Number of epochs"
    ),
    batch_size: int = typer.Option(
        4, "--batch-size", "-b", help="Per-device batch size"
    ),
    max_steps: int = typer.Option(
        20, "--max-steps", help="Maximum proof steps"
    ),
    reward_type: str = typer.Option(
        "shaped", "--reward", "-r",
        help="Reward config: binary, shaped, lenient, strict"
    ),
    use_mock_lean: bool = typer.Option(
        False, "--mock-lean",
        help="Use mock Lean interface (for testing)"
    ),
    use_mock_inference: bool = typer.Option(
        False, "--mock-inference",
        help="Use mock inference client (for testing)"
    ),
    lora_rank: int = typer.Option(
        8, "--lora-rank", help="LoRA rank"
    ),
    use_pipeline: bool = typer.Option(
        False, "--pipeline",
        help="Use iterative pipeline instead of single training run"
    ),
    iterations: int = typer.Option(
        10, "--iterations", "-i",
        help="Number of iterations for pipeline mode"
    ),
    algorithm_config: str = typer.Option(
        None, "--algo-config",
        help="JSON string with algorithm-specific config"
    ),
):
    """Train a model using GRPO on Lean 4 proofs."""
    algo_display = algorithm.upper()
    if algorithm.lower() == "drgrpo":
        algo_display = "Dr. GRPO (GRPO Done Right)"
    
    console.print(Panel.fit(
        f"[bold blue]Lean GRPO Training[/bold blue]\n"
        f"Algorithm: [green]{algo_display}[/green]\n"
        f"Model: {base_model}"
    ))
    
    # Validate algorithm
    if algorithm.lower() not in ALGORITHMS:
        console.print(f"[red]Error: Unknown algorithm '{algorithm}'. Choose from: {', '.join(ALGORITHMS)}[/red]")
        raise typer.Exit(1)
    
    # Load theorems
    with console.status("[bold green]Loading theorems..."):
        theorems = load_theorems(data_path)
    console.print(f"Loaded [bold]{len(theorems)}[/bold] theorems")
    
    # Create reward config
    reward_config = REWARD_CONFIGS.get(reward_type, REWARD_CONFIG_SHAPED)
    
    # Parse algorithm config
    algo_config = {}
    if algorithm_config:
        try:
            algo_config = json.loads(algorithm_config)
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON in --algo-config[/red]")
            raise typer.Exit(1)
    
    # Create configuration
    config = LeanGRPOConfig(
        base_model=base_model,
        algorithm=algorithm.lower(),
        algorithm_config=algo_config,
        lora_rank=lora_rank,
        num_generations=num_generations,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        max_proof_steps=max_steps,
        reward_config=reward_config,
        output_dir=str(output_dir),
    )
    
    # Create interfaces
    lean = MockLeanInterface() if use_mock_lean else LeanInterface()
    
    if use_mock_inference:
        inference = MockInferenceClient()
    else:
        api_key = inference_key or os.environ.get("OPENAI_API_KEY", "dummy-key")
        inference = VLLMClient(
            base_url=inference_url,
            api_key=api_key,
        )
    
    # Run training
    if use_pipeline:
        console.print("\n[bold]Running iterative training pipeline...[/bold]")
        pipeline = LeanGRPOPipeline(config, inference, lean)
        asyncio.run(pipeline.run_training_loop(
            theorems=theorems,
            num_iterations=iterations,
        ))
    else:
        console.print("\n[bold]Running single training...[/bold]")
        trainer = LeanGRPOTrainer(config, lean, inference)
        trainer.setup()
        
        # Prepare dataset
        with console.status("[bold green]Preparing dataset..."):
            dataset = trainer.prepare_dataset(theorems, trainer.tokenizer)
        
        console.print(f"Dataset size: [bold]{len(dataset)}[/bold]")
        
        # Train
        trainer.train(dataset)
        
        # Save
        trainer.save_model()
        console.print(f"\n[bold green]Model saved to {output_dir}[/bold green]")


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    data_path: Path = typer.Argument(..., help="Path to test theorems"),
    inference_url: str = typer.Option(
        "http://localhost:8000/v1", "--inference-url", help="Inference API URL"
    ),
    num_samples: int = typer.Option(
        10, "--num-samples", "-n", help="Number of samples per theorem"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Sampling temperature"
    ),
    use_mock_lean: bool = typer.Option(
        False, "--mock-lean", help="Use mock Lean interface"
    ),
):
    """Evaluate a trained model on test theorems."""
    console.print(Panel.fit(
        "[bold blue]Lean GRPO Evaluation[/bold blue]\n"
        f"Evaluating {model_path} on {data_path}"
    ))
    
    # Load theorems
    theorems = load_theorems(data_path)
    console.print(f"Loaded [bold]{len(theorems)}[/bold] theorems")
    
    # TODO: Implement evaluation logic
    console.print("[yellow]Evaluation not yet implemented[/yellow]")


@app.command()
def generate(
    theorem: str = typer.Argument(..., help="Theorem statement to prove"),
    model_path: Path = typer.Option(
        None, "--model", "-m", help="Path to trained model"
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model", help="Base model if no trained model"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Sampling temperature"
    ),
    max_tokens: int = typer.Option(
        512, "--max-tokens", help="Maximum tokens to generate"
    ),
    context: str = typer.Option(
        "", "--context", "-c", help="Additional context"
    ),
):
    """Generate a proof for a given theorem."""
    console.print(Panel.fit(
        "[bold blue]Lean Proof Generation[/bold blue]"
    ))
    
    # Setup model
    model_name = str(model_path) if model_path else base_model
    
    with console.status(f"[bold green]Loading model {model_name}..."):
        config = LeanGRPOConfig(base_model=model_name)
        trainer = LeanGRPOTrainer(config)
        trainer.setup()
    
    console.print("[bold]Theorem:[/bold]")
    console.print(f"  {theorem}")
    
    # Generate
    with console.status("[bold green]Generating proof..."):
        proof = trainer.generate_proof(
            theorem_statement=theorem,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    console.print("\n[bold]Generated Proof:[/bold]")
    console.print(Panel(proof, title="Proof", border_style="green"))


@app.command()
def check(
    lean_file: Path = typer.Argument(..., help="Path to Lean 4 file"),
    lean_cmd: str = typer.Option("lake", "--lean-cmd", help="Lean command"),
):
    """Check a Lean 4 file for errors."""
    console.print(Panel.fit(
        "[bold blue]Lean 4 File Check[/bold blue]"
    ))
    
    lean = LeanInterface(lean_cmd=lean_cmd)
    
    # Read file
    with open(lean_file, 'r') as f:
        content = f.read()
    
    console.print(f"Checking [bold]{lean_file}[/bold]...")
    
    # Create a simple state to check
    from lean_grpo.lean_interface import LeanProofState, TacticResult, TacticStatus
    
    # TODO: Parse file and check
    console.print("[yellow]File checking not yet fully implemented[/yellow]")


@app.command()
def stats(
    data_path: Path = typer.Argument(..., help="Path to theorem dataset"),
):
    """Show statistics about a theorem dataset."""
    console.print(Panel.fit(
        "[bold blue]Dataset Statistics[/bold blue]"
    ))
    
    theorems = load_theorems(data_path)
    
    # Create statistics table
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Theorems", str(len(theorems)))
    
    # Calculate average statement length
    if theorems:
        avg_length = sum(len(t.get("statement", "")) for t in theorems) / len(theorems)
        table.add_row("Avg Statement Length", f"{avg_length:.1f} chars")
    
    console.print(table)


@app.command()
def algorithms():
    """Show information about available RL algorithms."""
    console.print(Panel.fit(
        "[bold blue]Available RL Algorithms[/bold blue]"
    ))
    
    table = Table(title="Algorithm Details")
    table.add_column("Algorithm", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Best For", style="magenta")
    
    table.add_row(
        "GRPO",
        "Group Relative Policy Optimization. Group-normalized advantages, no value function.",
        "General use, stable training"
    )
    table.add_row(
        "DGPO",
        "Direct GRPO. Preference learning with DPO-style direct optimization.",
        "Pairwise preferences"
    )
    table.add_row(
        "DrGRPO",
        "Dr. GRPO (GRPO Done Right). Fixes for IS, KL estimation, and normalization.",
        "When GRPO has stability issues"
    )
    table.add_row(
        "DAPO",
        "Decoupled Advantage Policy Optimization. Population stats, asymmetric loss.",
        "Sparse/diverse rewards"
    )
    table.add_row(
        "GSPO",
        "Group-Synchronized Policy Optimization. Cross-group sync and consensus.",
        "Large-scale distributed"
    )
    
    console.print(table)
    
    console.print("\n[bold]Key Recommendation:[/bold]")
    console.print("  • Start with [cyan]GRPO[/cyan] for general use")
    console.print("  • Use [cyan]DrGRPO[/cyan] if you encounter stability issues")
    console.print("  • Use [cyan]DGPO[/cyan] if you have pairwise preferences")
    console.print("  • Use [cyan]DAPO[/cyan] for sparse rewards")
    console.print("  • Use [cyan]GSPO[/cyan] for distributed training")
    
    console.print("\n[bold]Usage:[/bold]")
    console.print("  lean-grpo train data.jsonl --algorithm grpo")
    console.print("  lean-grpo train data.jsonl --algorithm drgrpo")
    console.print("  lean-grpo train data.jsonl --algorithm dgpo --algo-config '{\"use_dpo_loss\": true}'")


def load_theorems(path: Path) -> list[dict]:
    """Load theorems from JSON or JSONL file."""
    theorems = []
    
    with open(path, 'r') as f:
        if path.suffix == '.jsonl':
            for line in f:
                if line.strip():
                    theorems.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                theorems = data
            elif isinstance(data, dict) and 'theorems' in data:
                theorems = data['theorems']
            else:
                theorems = [data]
    
    return theorems


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
