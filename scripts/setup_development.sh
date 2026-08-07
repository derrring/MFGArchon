#!/bin/bash
"""
Modern development environment setup script for MFGarchon.
Supports both UV and traditional pip/conda workflows.
"""

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
PROJECT_NAME="MFGarchon"
VENV_NAME="mfgarchon_dev"
PYTHON_VERSION="3.12"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup UV-based environment (preferred)
setup_uv_environment() {
    log_info "Setting up UV-based development environment..."

    if ! command_exists uv; then
        log_warning "UV not found. Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    fi

    # Create virtual environment with UV
    log_info "Creating Python ${PYTHON_VERSION} virtual environment..."
    uv venv --python "${PYTHON_VERSION}" .venv

    # Activate environment
    source .venv/bin/activate

    # Install package in development mode
    log_info "Installing MFGarchon in development mode..."
    uv pip install -e ".[dev]"

    # Install pre-commit hooks
    if command_exists pre-commit; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
    fi

    log_success "UV environment setup complete!"
    log_info "Activate with: source .venv/bin/activate"
}

# Setup traditional pip-based environment (fallback)
setup_pip_environment() {
    log_info "Setting up pip-based development environment..."

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install package in development mode
    log_info "Installing MFGarchon in development mode..."
    pip install -e ".[dev]"

    # Install pre-commit hooks
    if command_exists pre-commit; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
    fi

    log_success "Pip environment setup complete!"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Test basic import
    python -c "import mfgarchon; print(f'✅ MFGarchon version: {getattr(mfgarchon, \"__version__\", \"dev\")}')" || {
        log_error "Package import failed"
        return 1
    }

    # Check development tools
    if command_exists mypy; then
        log_success "MyPy available"
    else
        log_warning "MyPy not available"
    fi

    if command_exists ruff; then
        log_success "Ruff available"
    else
        log_warning "Ruff not available"
    fi

    # Run quick verification if script exists
    if [ -f "scripts/verify_modernization.py" ]; then
        log_info "Running modernization verification..."
        python scripts/verify_modernization.py || log_warning "Verification script had issues"
    fi
}

# Main setup function
main() {
    log_info "🚀 Setting up ${PROJECT_NAME} development environment"
    echo "=============================================="

    # Check if we're in the project root
    if [ ! -f "pyproject.toml" ]; then
        log_error "Must run from project root directory (containing pyproject.toml)"
        exit 1
    fi

    # Choose setup method
    if command_exists uv; then
        log_info "UV detected - using modern UV-based setup"
        setup_uv_environment
    else
        log_warning "UV not found - using traditional pip setup"
        setup_pip_environment
    fi

    # Verify everything works
    verify_installation

    echo ""
    log_success "🎉 Development environment ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: source .venv/bin/activate"
    echo "  2. Run quick type check: python scripts/quick_type_check.py"
    echo "  3. Run tests: python -m pytest"
    echo "  4. Happy coding! 🚀"
}

# Handle script arguments
case "${1:-setup}" in
    "setup"|"")
        main
        ;;
    "verify")
        verify_installation
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [setup|verify|help]"
        echo ""
        echo "Commands:"
        echo "  setup   - Set up development environment (default)"
        echo "  verify  - Verify existing installation"
        echo "  help    - Show this help message"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
