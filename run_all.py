#!/usr/bin/env python3
# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the "License");
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/axiom/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run all gameworld games with AXIOM.

This script runs AXIOM on all available gameworld games and logs results to wandb.
It automatically discovers available games and runs each one sequentially.

Usage:
    python run_all.py [--num_steps NUM_STEPS] [--parallel] [--games GAME1,GAME2,...]

Examples:
    python run_all.py                          # Run all games with default settings
    python run_all.py --num_steps 5000         # Run all games for 5000 steps each
    python run_all.py --games Explode,Bounce   # Run only specific games
    python run_all.py --parallel               # Run games in parallel (experimental)
"""

import os
import sys
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from datetime import datetime

# Import configuration optimizations
try:
    from config_long_runs import PRESET_CONFIGS, get_long_run_args
except ImportError:
    PRESET_CONFIGS = {}
    get_long_run_args = lambda: []

# Ensure wandb API key is set
if "WANDB_API_KEY" not in os.environ:
    os.environ["WANDB_API_KEY"] = "341793837021869d6d397352ab1caf45df6bdcc9"

# Available games discovered from gameworld.envs (verified complete list)
AVAILABLE_GAMES = [
    "Aviate", "Bounce", "Cross", "Drive", "Explode", 
    "Fruits", "Gold", "Hunt", "Impact", "Jump"
]

console = Console()


def run_single_game(game_name, num_steps=10000, additional_args=None, timeout_hours=2):
    """
    Run AXIOM on a single game.
    
    Args:
        game_name (str): Name of the game to run
        num_steps (int): Number of steps to run
        additional_args (list): Additional command line arguments
        timeout_hours (int): Timeout in hours for the run
        
    Returns:
        tuple: (game_name, success, execution_time, output)
    """
    start_time = time.time()
    
    try:
        # Build command
        cmd = [
            sys.executable, "main.py",
            f"--game={game_name}",
            f"--num_steps={num_steps}",
            f"--name=axiom-all-games",
            f"--group=run-all-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ]
        
        if additional_args:
            cmd.extend(additional_args)
            
        console.print(f"[bold blue]Starting {game_name}[/bold blue] with {num_steps} steps")
        console.print(f"[dim]Command: {' '.join(cmd[:6])}... (+{len(cmd)-6} more args)[/dim]")
        
        # Calculate timeout based on number of steps (longer runs need more time)
        base_timeout = timeout_hours * 3600  # Convert to seconds
        if num_steps > 50000:
            # Scale timeout for longer runs
            scale_factor = max(1.0, num_steps / 50000)
            timeout = int(base_timeout * scale_factor)
        else:
            timeout = base_timeout
            
        console.print(f"[dim]Timeout set to {timeout//3600:.1f} hours[/dim]")
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            console.print(f"[bold green]✓ {game_name} completed successfully[/bold green] in {execution_time:.1f}s")
            return game_name, True, execution_time, result.stdout
        else:
            console.print(f"[bold red]✗ {game_name} failed[/bold red] with return code {result.returncode}")
            console.print(f"[red]STDERR:[/red] {result.stderr}")
            return game_name, False, execution_time, result.stderr
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        console.print(f"[bold yellow]⚠ {game_name} timed out[/bold yellow] after {execution_time:.1f}s")
        return game_name, False, execution_time, "Timeout"
        
    except Exception as e:
        execution_time = time.time() - start_time
        console.print(f"[bold red]✗ {game_name} failed with exception[/bold red]: {e}")
        return game_name, False, execution_time, str(e)


def run_games_sequential(games, num_steps, additional_args, timeout_hours=2):
    """Run games sequentially with progress tracking."""
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        main_task = progress.add_task("Running all games...", total=len(games))
        
        for i, game in enumerate(games):
            progress.update(main_task, description=f"Running {game} ({i+1}/{len(games)})")
            
            result = run_single_game(game, num_steps, additional_args, timeout_hours)
            results.append(result)
            
            progress.advance(main_task)
    
    return results


def run_games_parallel(games, num_steps, additional_args, max_workers=None, timeout_hours=2):
    """Run games in parallel using ProcessPoolExecutor."""
    if max_workers is None:
        max_workers = min(len(games), mp.cpu_count() // 2)  # Conservative parallelism
    
    console.print(f"[bold blue]Running {len(games)} games in parallel with {max_workers} workers[/bold blue]")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        main_task = progress.add_task("Running games in parallel...", total=len(games))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_game = {
                executor.submit(run_single_game, game, num_steps, additional_args, timeout_hours): game 
                for game in games
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_game):
                game = future_to_game[future]
                try:
                    result = future.result()
                    results.append(result)
                    progress.advance(main_task)
                except Exception as e:
                    console.print(f"[bold red]✗ {game} failed with exception[/bold red]: {e}")
                    results.append((game, False, 0, str(e)))
                    progress.advance(main_task)
    
    return results


def print_summary(results):
    """Print a summary table of all results."""
    console.print("\n[bold]Summary of Results[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Game", style="dim", width=12)
    table.add_column("Status", justify="center")
    table.add_column("Time (s)", justify="right")
    table.add_column("Notes")
    
    successful = 0
    total_time = 0
    
    for game_name, success, execution_time, output in results:
        total_time += execution_time
        
        if success:
            successful += 1
            status = "[green]✓ Success[/green]"
            notes = "Completed successfully"
        else:
            status = "[red]✗ Failed[/red]"
            # Truncate error message for table
            notes = output[:50] + "..." if len(output) > 50 else output
            
        table.add_row(
            game_name,
            status,
            f"{execution_time:.1f}",
            notes
        )
    
    console.print(table)
    
    # Summary statistics
    console.print(f"\n[bold]Overall Statistics:[/bold]")
    console.print(f"  • Successful: {successful}/{len(results)} games")
    console.print(f"  • Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    console.print(f"  • Average time per game: {total_time/len(results):.1f} seconds")
    
    if successful == len(results):
        console.print(f"[bold green]🎉 All games completed successfully![/bold green]")
    elif successful > 0:
        console.print(f"[bold yellow]⚠ {len(results) - successful} games failed[/bold yellow]")
    else:
        console.print(f"[bold red]❌ All games failed[/bold red]")


def main():
    parser = argparse.ArgumentParser(
        description="Run AXIOM on all gameworld games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--games",
        type=str,
        default=None,
        help=f"Comma-separated list of games to run. Available: {', '.join(AVAILABLE_GAMES)}"
    )
    
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Number of steps to run each game for (default: 10000)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run games in parallel (experimental, may use more resources)"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (only used with --parallel)"
    )
    
    parser.add_argument(
        "--config_preset",
        type=str,
        choices=list(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else [],
        default=None,
        help=f"Use optimized configuration preset. Available: {', '.join(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else 'None (install config_long_runs.py)'}"
    )
    
    parser.add_argument(
        "--timeout_hours",
        type=int,
        default=2,
        help="Timeout in hours for each game run (automatically scaled for longer runs)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be run without actually running"
    )
    
    # Allow passing additional arguments to main.py
    parser.add_argument(
        "additional_args",
        nargs="*",
        help="Additional arguments to pass to main.py"
    )
    
    args = parser.parse_args()
    
    # Handle configuration presets
    preset_args = []
    if args.config_preset and PRESET_CONFIGS:
        if args.config_preset in PRESET_CONFIGS:
            preset_func = PRESET_CONFIGS[args.config_preset]["args_func"]
            preset_args = preset_func()
            console.print(f"[bold blue]Using config preset:[/bold blue] {args.config_preset}")
            console.print(f"[dim]{PRESET_CONFIGS[args.config_preset]['description']}[/dim]")
            
            # Extract num_steps from preset if not explicitly provided by user
            preset_num_steps_found = False
            for arg in preset_args:
                if arg.startswith("--num_steps="):
                    preset_num_steps = int(arg.split("=")[1])
                    if args.num_steps == 10000:  # Default value, user didn't specify
                        args.num_steps = preset_num_steps
                    preset_num_steps_found = True
                    break
            
            # Remove num_steps from preset args to avoid duplication
            if preset_num_steps_found:
                preset_args = [arg for arg in preset_args if not arg.startswith("--num_steps=")]
        else:
            console.print(f"[bold red]Error:[/bold red] Unknown preset '{args.config_preset}'")
            console.print(f"Available presets: {', '.join(PRESET_CONFIGS.keys())}")
            sys.exit(1)
    
    # Determine which games to run
    if args.games:
        games = [game.strip() for game in args.games.split(",")]
        # Validate game names
        invalid_games = [game for game in games if game not in AVAILABLE_GAMES]
        if invalid_games:
            console.print(f"[bold red]Error:[/bold red] Invalid games: {', '.join(invalid_games)}")
            console.print(f"Available games: {', '.join(AVAILABLE_GAMES)}")
            sys.exit(1)
    else:
        games = AVAILABLE_GAMES.copy()
    
    # Combine preset args with additional args
    combined_additional_args = preset_args + (args.additional_args or [])
    
    console.print(f"[bold]AXIOM Multi-Game Runner[/bold]")
    console.print(f"Games to run: {', '.join(games)}")
    console.print(f"Steps per game: {args.num_steps}")
    console.print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    if preset_args:
        console.print(f"Optimization preset: {args.config_preset}")
        console.print(f"Additional args: {len(combined_additional_args)} arguments")
    
    if args.dry_run:
        console.print(f"\n[bold yellow]DRY RUN - Commands that would be executed:[/bold yellow]")
        for game in games:
            cmd = [
                sys.executable, "main.py",
                f"--game={game}",
                f"--num_steps={args.num_steps}",
                f"--name=axiom-all-games",
                f"--group=run-all-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ]
            if combined_additional_args:
                cmd.extend(combined_additional_args)
            console.print(f"  {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
        
        if preset_args:
            console.print(f"\n[bold blue]Preset optimization arguments:[/bold blue]")
            console.print(f"  {' '.join(preset_args)}")
        return
    
    # Confirm before running
    if len(games) > 3:  # Only confirm for large runs
        if not console.input(f"\nAbout to run {len(games)} games. Continue? [y/N]: ").lower().startswith('y'):
            console.print("Cancelled.")
            return
    
    start_time = time.time()
    console.print(f"\n[bold blue]Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold blue]")
    
    try:
        if args.parallel:
            results = run_games_parallel(games, args.num_steps, combined_additional_args, args.max_workers, args.timeout_hours)
        else:
            results = run_games_sequential(games, args.num_steps, combined_additional_args, args.timeout_hours)
        
        total_time = time.time() - start_time
        console.print(f"\n[bold blue]Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold blue]")
        console.print(f"[bold blue]Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)[/bold blue]")
        
        print_summary(results)
        
        # Save detailed results to file
        results_file = f"run_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write(f"AXIOM Multi-Game Run Results\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Games: {', '.join(games)}\n")
            f.write(f"Steps per game: {args.num_steps}\n")
            f.write(f"Mode: {'Parallel' if args.parallel else 'Sequential'}\n\n")
            
            for game_name, success, execution_time, output in results:
                f.write(f"=== {game_name} ===\n")
                f.write(f"Success: {success}\n")
                f.write(f"Time: {execution_time:.1f}s\n")
                f.write(f"Output/Error:\n{output}\n\n")
        
        console.print(f"\n[dim]Detailed results saved to: {results_file}[/dim]")
        
    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]⚠ Interrupted by user[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main() 