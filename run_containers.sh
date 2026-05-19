#!/bin/bash

# LLM Tensor Server - Enhanced Container Management Script
# Comprehensive management for images and containers with Podman

set -e

# Colors for output
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

# Function to check if podman is available
check_podman() {
    if ! command -v podman &> /dev/null; then
        print_error "Podman is not installed. Please install Podman first."
        exit 1
    fi
}

# Function to show menu
show_menu() {
    echo
    echo "=========================================="
    echo "  LLM Tensor Server - Container Manager  "
    echo "=========================================="
    echo
    echo "📦 IMAGE MANAGEMENT:"
    echo "1.  Build all images"
    echo "2.  Build specific image"
    echo "3.  List all images"
    echo "4.  Remove specific image"
    echo "5.  Remove all LLM images"
    echo
    echo "🚀 CONTAINER MANAGEMENT:"
    echo "6.  Start all containers"
    echo "7.  Start specific container"
    echo "8.  Stop all containers"
    echo "9.  Stop specific container"
    echo "10. Restart all containers"
    echo "11. List all containers"
    echo "12. Remove specific container"
    echo "13. Remove all LLM containers"
    echo
    echo "📊 MONITORING:"
    echo "14. View container logs"
    echo "15. Show container status"
    echo
    echo "🧹 CLEANUP:"
    echo "16. Full cleanup (containers + images)"
    echo
    echo "17. Exit"
    echo
}

# Function to handle container run with potential name conflicts
run_container_with_replace() {
    local container_name="$1"
    local run_command="$2"
    
    # Try to run the container
    if eval "$run_command" 2>/dev/null; then
        print_success "Container $container_name started successfully!"
        return 0
    else
        # Check if it's a name conflict error
        if podman ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
            print_warning "Container '$container_name' already exists."
            echo
            echo "Options:"
            echo "1. Replace existing container (stop and remove old one)"
            echo "2. Skip this container"
            echo
            read -p "Choose option [1-2]: " choice
            
            case $choice in
                1)
                    print_status "Stopping and removing existing container..."
                    podman stop "$container_name" 2>/dev/null || true
                    podman rm "$container_name" 2>/dev/null || true
                    
                    print_status "Starting new container..."
                    if eval "$run_command"; then
                        print_success "Container $container_name replaced and started successfully!"
                        return 0
                    else
                        print_error "Failed to start replacement container"
                        return 1
                    fi
                    ;;
                2)
                    print_status "Skipping $container_name"
                    return 0
                    ;;
                *)
                    print_error "Invalid option. Skipping $container_name"
                    return 1
                    ;;
            esac
        else
            print_error "Failed to start container $container_name (unknown error)"
            return 1
        fi
    fi
}

# Function to build all images
build_all() {
    print_status "Building all microservice images..."
    
    cd containers/
    
    print_status "Building orchestrator image..."
    podman build -f Containerfile.orchestrator -t llm-orchestrator:latest ..
    
    print_status "Building Transformers image..."
    podman build -f Containerfile.transformers -t llm-transformers:latest ..
    
    print_status "Building vLLM image..."
    podman build -f Containerfile.vllm -t llm-vllm:latest ..
    
    print_status "Building TensorRT image..."
    podman build -f Containerfile.tensorrt -t llm-tensorrt:latest ..

    print_status "Building TensorRT Builder image..."
    podman build -f Containerfile.tensorrt-builder -t llm-tensorrt-builder:latest ..

    cd ..
    print_success "All images built successfully!"
}

# Function to build specific image
build_specific() {
    echo
    echo "Available images to build:"
    echo "1. Orchestrator"
    echo "2. Transformers"
    echo "3. vLLM"
    echo "4. TensorRT"
    echo "5. TensorRT Builder"
    echo
    read -p "Select image to build (1-5): " choice
    
    cd containers/
    
    case $choice in
        1)
            print_status "Building orchestrator image..."
            podman build -f Containerfile.orchestrator -t llm-orchestrator:latest ..
            ;;
        2)
            print_status "Building Transformers image..."
            podman build -f Containerfile.transformers -t llm-transformers:latest ..
            ;;
        3)
            print_status "Building vLLM image..."
            podman build -f Containerfile.vllm -t llm-vllm:latest ..
            ;;
        4)
            print_status "Building TensorRT image..."
            podman build -f Containerfile.tensorrt -t llm-tensorrt:latest ..
            ;;
        5)
            print_status "Building TensorRT Builder image..."
            podman build -f Containerfile.tensorrt-builder -t llm-tensorrt-builder:latest ..
            ;;
        *)
            print_error "Invalid selection"
            cd ..
            return 1
            ;;
    esac
    
    cd ..
    print_success "Image built successfully!"
}

# Function to list LLM images
list_images() {
    print_status "LLM Tensor Server images:"
    local llm_images=$(podman images --format "{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" | grep -E "(llm-orchestrator|llm-transformers|llm-vllm|llm-tensorrt)")
    
    if [ -n "$llm_images" ]; then
        echo "Repository          Tag     Image ID      Size"
        echo "==================  ======  ============  ========"
        echo "$llm_images"
    else
        echo "No LLM Tensor Server images found"
        echo "Run 'Build all images' or 'Build specific image' first"
    fi
}

# Function to remove specific image
remove_image() {
    list_images
    echo
    echo "Available images to remove:"
    echo "1. llm-orchestrator"
    echo "2. llm-transformers"
    echo "3. llm-vllm"
    echo "4. llm-tensorrt"
    echo
    read -p "Select image to remove (1-4): " choice
    
    local image_name=""
    case $choice in
        1) image_name="llm-orchestrator" ;;
        2) image_name="llm-transformers" ;;
        3) image_name="llm-vllm" ;;
        4) image_name="llm-tensorrt" ;;
        *)
            print_error "Invalid selection"
            return 1
            ;;
    esac
    
    print_warning "This will remove the $image_name image. Are you sure? (y/N)"
    read -r confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        print_status "Removing $image_name image..."
        podman rmi "$image_name:latest" 2>/dev/null || true
        print_success "$image_name image removed!"
    else
        print_status "Operation cancelled"
    fi
}

# Function to remove all LLM images
remove_all_images() {
    print_warning "This will remove ALL LLM Tensor Server images. Are you sure? (y/N)"
    read -r confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        print_status "Removing all LLM images..."
        podman rmi llm-orchestrator:latest llm-transformers:latest llm-vllm:latest llm-tensorrt:latest 2>/dev/null || true
        print_success "All LLM images removed!"
    else
        print_status "Operation cancelled"
    fi
}

# Function to start all containers
start_all() {
    print_status "Starting all microservice containers..."
    
    if command -v podman-compose &> /dev/null && [ -f containers/compose.yml ]; then
        cd containers/
        podman-compose -f compose.yml --profile full up -d
        cd ..
        print_success "All containers started with podman-compose!"
    else
        print_warning "podman-compose not found or compose.yml missing, starting manually..."
        
        # Create network
        podman network create llm-network 2>/dev/null || true
        
        # Start services with error handling
        run_container_with_replace "llm-transformers" "podman run -d --name llm-transformers --network llm-network -p 8012:8012 -v ./models:/app/models:Z --device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=0 llm-transformers:latest"
        
        run_container_with_replace "llm-vllm" "podman run -d --name llm-vllm --network llm-network -p 8013:8013 -v ./models:/app/models:Z --device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=1 llm-vllm:latest"
        
        run_container_with_replace "llm-tensorrt" "podman run -d --name llm-tensorrt --network llm-network -p 8014:8014 -v ./models:/app/models:Z --device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=0,1 llm-tensorrt:latest"
        
        run_container_with_replace "llm-orchestrator" "podman run -d --name llm-orchestrator --network llm-network -p 8011:8011 -v ./models:/app/models:Z -e TRANSFORMERS_SERVICE_URL=http://llm-transformers:8012 -e VLLM_SERVICE_URL=http://llm-vllm:8013 -e TENSORRT_SERVICE_URL=http://llm-tensorrt:8014 llm-orchestrator:latest"
        
        print_success "All containers started manually!"
    fi
}

# Function to start specific container
start_specific() {
    echo
    echo "Available containers to start:"
    echo "1. Orchestrator"
    echo "2. Transformers"
    echo "3. vLLM"
    echo "4. TensorRT"
    echo
    read -p "Select container to start (1-4): " choice
    
    # Create network if it doesn't exist
    podman network create llm-network 2>/dev/null || true
    
    case $choice in
        1)
            run_container_with_replace "llm-orchestrator" "podman run -d --name llm-orchestrator --network llm-network -p 8011:8011 -v ./models:/app/models:Z -e TRANSFORMERS_SERVICE_URL=http://llm-transformers:8012 -e VLLM_SERVICE_URL=http://llm-vllm:8013 -e TENSORRT_SERVICE_URL=http://llm-tensorrt:8014 llm-orchestrator:latest"
            ;;
        2)
            run_container_with_replace "llm-transformers" "podman run -d --name llm-transformers --network llm-network -p 8012:8012 -v ./models:/app/models:Z --device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=0 llm-transformers:latest"
            ;;
        3)
            run_container_with_replace "llm-vllm" "podman run -d --name llm-vllm --network llm-network -p 8013:8013 -v ./models:/app/models:Z --device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=1 llm-vllm:latest"
            ;;
        4)
            run_container_with_replace "llm-tensorrt" "podman run -d --name llm-tensorrt --network llm-network -p 8014:8014 -v ./models:/app/models:Z --device nvidia.com/gpu=all -e CUDA_VISIBLE_DEVICES=0,1 llm-tensorrt:latest"
            ;;
        *)
            print_error "Invalid selection"
            return 1
            ;;
    esac
}

# Function to stop all containers
stop_all() {
    print_status "Stopping all containers..."
    
    if command -v podman-compose &> /dev/null && [ -f containers/compose.yml ]; then
        cd containers/
        podman-compose -f compose.yml down
        cd ..
    else
        # Manual stop
        podman stop llm-orchestrator llm-transformers llm-vllm llm-tensorrt 2>/dev/null || true
    fi
    
    print_success "All containers stopped!"
}

# Function to stop specific container
stop_specific() {
    echo
    echo "Running containers:"
    podman ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(llm-orchestrator|llm-transformers|llm-vllm|llm-tensorrt)" || echo "No LLM containers currently running"
    echo
    read -p "Enter container name to stop: " container_name
    
    if [ -n "$container_name" ]; then
        if podman ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
            print_status "Stopping $container_name..."
            podman stop "$container_name"
            print_success "Container $container_name stopped!"
        else
            print_error "Container '$container_name' is not running"
        fi
    else
        print_error "No container name provided"
    fi
}

# Function to restart all containers
restart_all() {
    print_status "Restarting all containers..."
    stop_all
    sleep 2
    start_all
    print_success "All containers restarted!"
}

# Function to list all LLM containers
list_containers() {
    print_status "LLM Tensor Server containers:"
    
    local llm_containers=$(podman ps -a --format "{{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -E "(llm-orchestrator|llm-transformers|llm-vllm|llm-tensorrt)")
    
    if [ -n "$llm_containers" ]; then
        echo "Name                Status                      Ports"
        echo "==================  ========================    =========================="
        echo "$llm_containers"
    else
        echo "No LLM Tensor Server containers found"
        echo "Run 'Start all containers' or 'Start specific container' first"
    fi
}

# Function to remove specific container
remove_container() {
    list_containers
    echo
    read -p "Enter container name to remove: " container_name
    
    if [ -n "$container_name" ]; then
        if podman ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
            print_warning "This will stop and remove container '$container_name'. Are you sure? (y/N)"
            read -r confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                print_status "Stopping and removing $container_name..."
                podman stop "$container_name" 2>/dev/null || true
                podman rm "$container_name" 2>/dev/null || true
                print_success "Container $container_name removed!"
            else
                print_status "Operation cancelled"
            fi
        else
            print_error "Container '$container_name' not found"
        fi
    else
        print_error "No container name provided"
    fi
}

# Function to remove all LLM containers
remove_all_containers() {
    print_warning "This will stop and remove ALL LLM containers. Are you sure? (y/N)"
    read -r confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        print_status "Stopping and removing all LLM containers..."
        podman stop llm-orchestrator llm-transformers llm-vllm llm-tensorrt 2>/dev/null || true
        podman rm llm-orchestrator llm-transformers llm-vllm llm-tensorrt 2>/dev/null || true
        print_success "All LLM containers removed!"
    else
        print_status "Operation cancelled"
    fi
}

# Function to view logs
view_logs() {
    echo
    echo "All LLM Tensor Server containers:"
    
    local all_containers=$(podman ps -a --format "{{.Names}}\t{{.Status}}" 2>/dev/null | grep -E "(llm-orchestrator|llm-transformers|llm-vllm|llm-tensorrt)")
    
    if [ -n "$all_containers" ]; then
        echo "Name                Status"
        echo "==================  =========================="
        echo "$all_containers"
    else
        echo "No LLM Tensor Server containers found"
        echo
        echo "Available container names:"
        echo "  • llm-orchestrator"
        echo "  • llm-transformers" 
        echo "  • llm-vllm"
        echo "  • llm-tensorrt"
    fi
    
    echo
    read -p "Enter container name to view logs: " container_name
    
    if [ -n "$container_name" ]; then
        if podman ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
            print_status "Showing logs for $container_name (Press Ctrl+C to exit)..."
            podman logs -f "$container_name"
        else
            print_error "Container '$container_name' not found"
        fi
    else
        print_error "No container name provided"
    fi
}

# Function to show container status
show_status() {
    print_status "Current system status:"
    echo
    echo "=== IMAGES ==="
    list_images
    echo
    echo "=== CONTAINERS ==="
    list_containers
    echo
    echo "=== NETWORK ==="
    if podman network exists llm-network 2>/dev/null; then
        echo "Network 'llm-network': EXISTS"
    else
        echo "Network 'llm-network': NOT FOUND"
    fi
}

# Function for full cleanup
full_cleanup() {
    print_warning "This will remove ALL LLM containers AND images. Are you sure? (y/N)"
    read -r confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        print_status "Performing full cleanup..."
        
        # Stop and remove containers
        podman stop llm-orchestrator llm-transformers llm-vllm llm-tensorrt 2>/dev/null || true
        podman rm llm-orchestrator llm-transformers llm-vllm llm-tensorrt 2>/dev/null || true
        
        # Remove images
        podman rmi llm-orchestrator:latest llm-transformers:latest llm-vllm:latest llm-tensorrt:latest 2>/dev/null || true
        
        # Remove network
        podman network rm llm-network 2>/dev/null || true
        
        print_success "Full cleanup completed!"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main interactive loop
main() {
    print_status "LLM Tensor Server - Enhanced Container Manager"
    check_podman
    
    while true; do
        show_menu
        read -p "Select an option (1-17): " choice
        
        case $choice in
            1) build_all ;;
            2) build_specific ;;
            3) list_images ;;
            4) remove_image ;;
            5) remove_all_images ;;
            6) start_all ;;
            7) start_specific ;;
            8) stop_all ;;
            9) stop_specific ;;
            10) restart_all ;;
            11) list_containers ;;
            12) remove_container ;;
            13) remove_all_containers ;;
            14) view_logs ;;
            15) show_status ;;
            16) full_cleanup ;;
            17) 
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-17."
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"