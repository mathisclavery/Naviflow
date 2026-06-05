"""Affichage console colore + emojis pour le suivi des pipelines.

Couleurs ANSI : fonctionnent dans tout terminal moderne (bash, zsh, VS Code).
Pas de dependance externe (on evite colorama/rich pour rester leger sur la VM).
"""


class C:
    """Codes ANSI."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"


def title(msg):
    """Gros titre d'etape (encadre)."""
    bar = "=" * 60
    print(f"\n{C.BOLD}{C.CYAN}{bar}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {msg}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")


def step(msg):
    """Etape en cours (fleche bleue)."""
    print(f"{C.BLUE}▶  {msg}{C.RESET}")


def info(msg):
    """Information secondaire (grise)."""
    print(f"{C.DIM}   {msg}{C.RESET}")


def success(msg):
    """Succes (vert + check)."""
    print(f"{C.GREEN}✅ {msg}{C.RESET}")


def warn(msg):
    """Avertissement (jaune)."""
    print(f"{C.YELLOW}⚠️  {msg}{C.RESET}")


def progress(i, total, msg):
    """Ligne de progression d'une boucle (ex: [12/743])."""
    print(f"{C.MAGENTA}🚂 [{i}/{total}]{C.RESET} {msg}")


def done(msg):
    """Fin de pipeline (vert gras + drapeau)."""
    print(f"\n{C.BOLD}{C.GREEN}🏁 {msg}{C.RESET}\n")
