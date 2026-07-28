#!/bin/bash
# MFGarchon Multi-Environment Management Script
# Manages different conda environments for various use cases

set -e  # Exit on error

echo "🐍 MFGarchon Environment Manager"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Anaconda or Miniconda."
        exit 1
    fi
    print_status "Conda is available"
}

# Function to create development environment
create_dev_env() {
    print_info "Creating development environment (mfg_dev)..."

    conda env create -f environment.yml -n mfg_dev || {
        print_warning "Environment creation failed, trying update..."
        conda env update -f environment.yml -n mfg_dev
    }

    print_status "Activating environment and installing MFGarchon..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mfg_dev
    pip install -e .

    print_status "Development environment ready!"
}

# Function to create performance-optimized environment
create_performance_env() {
    print_info "Creating performance environment (mfg_performance)..."

    conda env create -f conda_performance.yml -n mfg_performance || {
        print_warning "Environment creation failed, trying update..."
        conda env update -f conda_performance.yml -n mfg_performance
    }

    print_status "Activating environment and installing MFGarchon..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mfg_performance
    pip install -e .

    print_status "Performance environment ready!"
}

# Function to create NumPy 2.0+ testing environment
create_numpy2_env() {
    print_info "Creating NumPy 2.0+ testing environment (mfg_numpy2)..."

    cat > /tmp/numpy2_env.yml << EOF
name: mfg_numpy2
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - "numpy>=2.0"
  - "scipy>=1.10"
  - matplotlib>=3.5
  - jupyter>=1.0
  - pytest>=6.0
  - pip
  - pip:
    - -e .
EOF

    conda env create -f /tmp/numpy2_env.yml || {
        print_warning "NumPy 2.0+ not available yet, creating preview environment..."

        conda create -n mfg_numpy2 python=3.12 scipy matplotlib jupyter pytest pip -y
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate mfg_numpy2
        pip install --pre --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy
        pip install -e .
    }

    print_status "NumPy 2.0+ testing environment ready!"
}

# Function to verify environments
verify_environments() {
    print_info "Verifying MFGarchon environments..."

    local envs=("mfg_dev" "mfg_performance" "mfg_numpy2")

    for env in "${envs[@]}"; do
        if conda env list | grep -q "^$env "; then
            print_status "Environment $env exists"

            # Test the environment
            source $(conda info --base)/etc/profile.d/conda.sh
            conda activate "$env"

            print_info "Testing $env environment..."
            python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

import numpy as np
print(f'NumPy: {np.__version__}')

try:
    import mfgarchon
    print(f'MFGarchon: {mfgarchon.__version__}')
    print('✅ MFGarchon import successful')
except Exception as e:
    print(f'❌ MFGarchon import failed: {e}')

try:
    from mfgarchon.utils.numpy_compat import get_numpy_info
    info = get_numpy_info()
    print(f'Recommended integration: {info[\"recommended_method\"]}')
except Exception as e:
    print(f'❌ NumPy compatibility check failed: {e}')
"
            echo "---"
        else
            print_warning "Environment $env not found"
        fi
    done
}

# Function to clean environments
clean_environments() {
    print_warning "This will remove all MFGarchon environments. Continue? (y/N)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        local envs=("mfg_dev" "mfg_performance" "mfg_numpy2")

        for env in "${envs[@]}"; do
            if conda env list | grep -q "^$env "; then
                print_info "Removing environment $env..."
                conda env remove -n "$env" -y
                print_status "Environment $env removed"
            fi
        done
    else
        print_info "Cleanup cancelled"
    fi
}

# Function to show environment info
show_env_info() {
    print_info "MFGarchon Environment Information"
    echo

    print_info "Available environments:"
    conda env list | grep mfg || print_warning "No MFGarchon environments found"

    echo
    print_info "Current environment:"
    echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-Not set}"

    if [[ -n "$CONDA_DEFAULT_ENV" ]] && [[ "$CONDA_DEFAULT_ENV" == mfg* ]]; then
        echo
        print_info "Current environment details:"
        # Removed with tests/verify_environment.py (#1716). The call named a bare
        # relative path the file never occupied, and `|| print_warning` swallowed the
        # resulting "file not found" -- so it reported a soft warning whether or not
        # anything ran. `python -c "import mfgarchon"` below is the real check.
        python -c "import mfgarchon" 2>/dev/null || print_warning "mfgarchon does not import in this environment"
    fi
}

# Function to activate environment with setup
activate_env() {
    local env_name="$1"

    if [[ -z "$env_name" ]]; then
        print_error "Please specify environment name (mfg_dev, mfg_performance, mfg_numpy2)"
        return 1
    fi

    if ! conda env list | grep -q "^$env_name "; then
        print_error "Environment $env_name not found"
        return 1
    fi

    print_info "Activating environment $env_name..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$env_name"

    # Source environment variables if available
    if [[ -f "setup_env_vars.sh" ]]; then
        print_info "Loading environment variables..."
        source setup_env_vars.sh
    fi

    print_status "Environment $env_name activated!"
    print_info "Run 'python -c \"import mfgarchon\"' to test the setup"
}

# Main menu
case "$1" in
    "create-dev")
        check_conda
        create_dev_env
        ;;
    "create-performance")
        check_conda
        create_performance_env
        ;;
    "create-numpy2")
        check_conda
        create_numpy2_env
        ;;
    "create-all")
        check_conda
        create_dev_env
        create_performance_env
        create_numpy2_env
        ;;
    "verify")
        check_conda
        verify_environments
        ;;
    "clean")
        check_conda
        clean_environments
        ;;
    "info")
        check_conda
        show_env_info
        ;;
    "activate")
        check_conda
        activate_env "$2"
        ;;
    *)
        echo "Usage: $0 {create-dev|create-performance|create-numpy2|create-all|verify|clean|info|activate <env_name>}"
        echo
        echo "Commands:"
        echo "  create-dev         Create development environment"
        echo "  create-performance Create performance-optimized environment"
        echo "  create-numpy2      Create NumPy 2.0+ testing environment"
        echo "  create-all         Create all environments"
        echo "  verify             Verify all environments"
        echo "  clean              Remove all MFGarchon environments"
        echo "  info               Show environment information"
        echo "  activate <name>    Activate specific environment"
        echo
        echo "Example:"
        echo "  $0 create-dev      # Create development environment"
        echo "  $0 activate mfg_dev    # Activate development environment"
        exit 1
        ;;
esac
