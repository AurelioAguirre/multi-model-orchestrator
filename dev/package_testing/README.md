# Import Testing for LLM Tensor Server

This directory contains systematic testing for package compatibility across the three ML frameworks used in the LLM Tensor Server.

## Purpose

The microservices architecture requires each framework to have its own isolated dependencies. This testing ensures:

1. **Package compatibility** - All packages install without conflicts
2. **Import functionality** - Core imports work correctly  
3. **Version freezing** - Generate working requirements with exact versions

## Structure

```
test/
├── run_import_tests.sh              # Main testing script
├── README.md                        # This file
└── import_tests/
    ├── transformers/
    │   ├── requirements.txt          # Unpinned requirements
    │   ├── import_test.py            # Import test script
    │   └── requirements_frozen.txt   # Generated after successful test
    ├── vllm/
    │   ├── requirements.txt          # Unpinned requirements
    │   ├── import_test.py            # Import test script
    │   ├── wheels/                   # Custom vLLM wheel
    │   └── requirements_frozen.txt   # Generated after successful test
    └── tensorrt/
        ├── requirements.txt          # Unpinned requirements (numpy<2)
        ├── import_test.py            # Import test script
        ├── wheels/                   # Custom TensorRT wheel
        └── requirements_frozen.txt   # Generated after successful test
```

## Usage

### Test All Frameworks
```bash
cd test/
chmod +x run_import_tests.sh
./run_import_tests.sh
```

### Test Individual Framework
```bash
# Test only transformers
./run_import_tests.sh transformers

# Test only vLLM
./run_import_tests.sh vllm

# Test only TensorRT
./run_import_tests.sh tensorrt
```

## Process

For each framework, the script will:

1. **Create virtual environment** using Python 3.12
2. **Install unpinned requirements** from `requirements.txt`
3. **Install custom wheels** (if available in `wheels/` directory)
4. **Run import tests** via `import_test.py`
5. **Freeze requirements** to `requirements_frozen.txt` if successful

## Import Tests

Each `import_test.py` performs:

- **Basic imports** - Core framework and dependencies
- **Version reporting** - Shows installed versions
- **Functionality tests** - Basic operations to ensure packages work
- **Compatibility checks** - Framework-specific requirements

## Using Results

After successful tests:

1. **Review frozen requirements** in `requirements_frozen.txt` files
2. **Copy to podman requirements** - Update `requirements/podman/` files
3. **Test container builds** - Verify containers build successfully
4. **Update documentation** - Note any version constraints discovered

## Troubleshooting

### Common Issues

**Import failures:**
- Check error messages in test output
- Verify Python 3.12 installation
- Ensure virtual environment is clean

**Version conflicts:**
- Review unpinned requirements for incompatibilities
- Adjust version constraints (e.g., `numpy<2` for TensorRT)
- Test frameworks in isolation

**Custom wheel issues:**
- Verify wheel files exist in `wheels/` directories
- Check wheel compatibility with Python 3.12
- Ensure wheel dependencies are satisfied

### Manual Testing

If automated tests fail, you can manually test:

```bash
cd test/import_tests/transformers
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python import_test.py
```

## Next Steps

After all tests pass:

1. **Update Podmanfiles** with frozen requirements
2. **Test container builds** to verify fixes
3. **Document version constraints** in project documentation
4. **Consider CI integration** for future changes