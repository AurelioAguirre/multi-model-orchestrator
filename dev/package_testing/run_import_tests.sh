#!/bin/bash

# Import Testing Script for LLM Tensor Server
# Tests package installations and imports for each framework

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$SCRIPT_DIR/import_tests"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

test_framework() {
    local framework=$1
    local test_path="$TEST_DIR/$framework"
    
    print_status "Testing $framework framework..."
    
    if [ ! -d "$test_path" ]; then
        print_error "Test directory not found: $test_path"
        return 1
    fi
    
    cd "$test_path"
    
    # Create virtual environment
    print_status "Creating virtual environment for $framework..."
    python3.12 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing requirements for $framework..."
    pip install -r requirements.txt
    
    # Install custom wheels if available
    if [ -d "wheels" ] && [ "$(ls -A wheels)" ]; then
        print_status "Installing custom wheels for $framework..."
        pip install wheels/*.whl --force-reinstall --no-deps
    fi
    
    # Run import test
    print_status "Running import test for $framework..."
    if python import_test.py; then
        print_success "$framework test passed!"
        
        # Freeze requirements
        print_status "Freezing requirements for $framework..."
        pip freeze > requirements_frozen.txt
        print_success "Frozen requirements saved to requirements_frozen.txt"
        
        deactivate
        return 0
    else
        print_error "$framework test failed!"
        deactivate
        return 1
    fi
}

main() {
    print_status "Starting import tests for all frameworks..."
    echo "This will create virtual environments and test package installations."
    echo ""
    
    # Check Python version
    if ! command -v python3.12 &> /dev/null; then
        print_error "Python 3.12 not found. Please install Python 3.12."
        exit 1
    fi
    
    frameworks=("transformers" "vllm" "tensorrt")
    failed_tests=()
    
    for framework in "${frameworks[@]}"; do
        echo ""
        echo "=" * 60
        echo "Testing $framework"
        echo "=" * 60
        
        if test_framework "$framework"; then
            print_success "$framework test completed successfully"
        else
            print_error "$framework test failed"
            failed_tests+=("$framework")
        fi
    done
    
    echo ""
    echo "=" * 60
    echo "Test Summary"
    echo "=" * 60
    
    if [ ${#failed_tests[@]} -eq 0 ]; then
        print_success "All tests passed! ✅"
        print_status "Frozen requirements available in each test directory."
        print_status "You can now use these for your Podmanfiles."
    else
        print_error "Some tests failed: ${failed_tests[*]}"
        print_status "Check the output above for details."
        exit 1
    fi
}

# Allow individual framework testing
if [ $# -eq 1 ]; then
    framework=$1
    if [[ "$framework" =~ ^(transformers|vllm|tensorrt)$ ]]; then
        test_framework "$framework"
        exit $?
    else
        print_error "Invalid framework: $framework"
        print_status "Usage: $0 [transformers|vllm|tensorrt]"
        exit 1
    fi
else
    main
fi