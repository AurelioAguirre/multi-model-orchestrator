#!/bin/bash

# LLM Tensor Server - Log Viewer Script
# Interactive log viewer for microservices containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  LLM Tensor Server - Log Viewer${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if podman is available
check_podman() {
    if ! command -v podman &> /dev/null; then
        print_error "Podman is not installed or not in PATH"
        exit 1
    fi
}

# Function to get running containers
get_running_containers() {
    local containers=()
    
    # Get all running container names (both running and recently stopped)
    local all_names=$(podman ps -a --format "{{.Names}}" 2>/dev/null)
    
    # Look for our LLM containers (using substring matching)
    local container_patterns=("llm-orchestrator" "llm-transformers" "llm-vllm" "llm-tensorrt")
    
    for pattern in "${container_patterns[@]}"; do
        local exact_name=$(echo "$all_names" | grep "$pattern" | head -1)
        if [ -n "$exact_name" ]; then
            containers+=("$exact_name")
        fi
    done
    
    echo "${containers[@]}"
}

# Function to display container menu and get selection
show_container_menu() {
    local containers=($1)
    local count=${#containers[@]}
    
    if [ $count -eq 0 ]; then
        print_error "No LLM Tensor Server containers are currently running"
        echo
        echo "Available containers to start:"
        echo "  • llm-orchestrator  (API gateway)"
        echo "  • llm-transformers  (HuggingFace Transformers service)"
        echo "  • llm-vllm          (vLLM high-performance service)"
        echo "  • llm-tensorrt      (TensorRT-LLM optimized service)"
        echo
        echo "Use './run_containers.sh' to start services"
        exit 1
    fi
    
    print_info "Found $count running container(s):"
    echo
    
    for i in "${!containers[@]}"; do
        local num=$((i + 1))
        local name="${containers[i]}"
        local status=$(podman ps --filter "name=^${name}$" --format "{{.Status}}")
        local ports=$(podman ps --filter "name=^${name}$" --format "{{.Ports}}")
        
        echo -e "${GREEN}[$num]${NC} ${name}"
        echo "     Status: ${status}"
        if [ -n "$ports" ]; then
            echo "     Ports:  ${ports}"
        fi
        echo
    done
}

# Function to get user selection
get_user_selection() {
    local containers=($1)
    local count=${#containers[@]}
    
    while true; do
        printf "Select container to view logs [1-$count]: "
        read -r selection
        
        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "$count" ]; then
            local index=$((selection - 1))
            printf "${containers[index]}"
            return
        else
            print_error "Invalid selection. Please enter a number between 1 and $count"
        fi
    done
}

# Function to show logs
show_logs() {
    local container_name="$1"
    
    # Debug: show exactly what we received
    echo "DEBUG: Container name received: '$container_name'"
    echo "DEBUG: Container name length: ${#container_name}"
    
    # Clean the container name (extract just the container name part)
    container_name=$(echo "$container_name" | grep -o 'llm-[a-zA-Z]*' | head -1)
    
    echo "DEBUG: Cleaned container name: '$container_name'"
    
    print_info "Starting live log stream for: $container_name"
    print_info "Press Ctrl+C to stop..."
    echo
    echo -e "${YELLOW}==================== LOGS FOR $container_name ====================${NC}"
    echo
    
    # Follow logs with timestamps and colors
    podman logs -f --timestamps "$container_name" 2>&1 | while IFS= read -r line; do
        # Add color coding based on log level
        if [[ "$line" =~ ERROR|FATAL ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ "$line" =~ WARNING|WARN ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ "$line" =~ INFO ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ "$line" =~ DEBUG ]]; then
            echo -e "${BLUE}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# Main function
main() {
    print_header
    check_podman
    
    # Get running containers
    local running_containers=($(get_running_containers))
    
    # Show menu and get selection
    show_container_menu "${running_containers[*]}"
    local selected_container=$(get_user_selection "${running_containers[*]}")
    
    echo
    # Show logs for selected container
    show_logs "$selected_container"
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}[INFO]${NC} Log viewing stopped. Goodbye!"; exit 0' SIGINT

# Run main function
main "$@"