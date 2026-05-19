#!/bin/bash

# kuberun.sh - Kubernetes Management Script for LLM Tensor Server
# Quick access to common kubectl operations

NAMESPACE="llm-inference"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to check which inference engine is running
check_running_engine() {
    echo -e "${BLUE}Checking inference engines...${NC}"

    TRANSFORMERS_REPLICAS=$(kubectl get deployment llm-transformers -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    VLLM_REPLICAS=$(kubectl get deployment llm-vllm -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    TENSORRT_REPLICAS=$(kubectl get deployment llm-tensorrt -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

    echo "  Transformers: $TRANSFORMERS_REPLICAS replica(s)"
    echo "  vLLM:         $VLLM_REPLICAS replica(s)"
    echo "  TensorRT:     $TENSORRT_REPLICAS replica(s)"
    echo ""
}

# Function to switch inference engines
switch_engine() {
    local target_engine=$1

    print_header "Switching to $target_engine"

    # Scale down all engines
    echo "Scaling down all inference engines..."
    kubectl scale deployment llm-transformers -n $NAMESPACE --replicas=0
    kubectl scale deployment llm-vllm -n $NAMESPACE --replicas=0
    kubectl scale deployment llm-tensorrt -n $NAMESPACE --replicas=0

    # Wait for pods to terminate
    echo "Waiting for pods to terminate..."
    sleep 5

    # Scale up target engine
    echo "Scaling up $target_engine..."
    case $target_engine in
        transformers)
            kubectl scale deployment llm-transformers -n $NAMESPACE --replicas=1
            ;;
        vllm)
            kubectl scale deployment llm-vllm -n $NAMESPACE --replicas=1
            ;;
        tensorrt)
            kubectl scale deployment llm-tensorrt -n $NAMESPACE --replicas=1
            ;;
    esac

    # Wait and check status
    echo "Waiting for pod to start..."
    sleep 10
    kubectl get pods -n $NAMESPACE

    print_success "$target_engine engine is starting up"
}

# Function to view logs
view_logs() {
    local component=$1
    local follow=$2

    print_header "Viewing logs: $component"

    if [ "$follow" = "follow" ]; then
        kubectl logs -l app=$component -n $NAMESPACE -f
    else
        kubectl logs -l app=$component -n $NAMESPACE --tail=50
    fi
}

# Function to port-forward
port_forward() {
    local service=$1
    local port=$2

    print_header "Port-forwarding $service on port $port"
    print_warning "Press Ctrl+C to stop port-forwarding"

    kubectl port-forward -n $NAMESPACE svc/$service $port:$port
}

# Main menu
show_menu() {
    clear
    print_header "LLM Tensor Server - Kubernetes Manager"

    echo -e "${GREEN}Current Status:${NC}"
    kubectl get pods -n $NAMESPACE 2>/dev/null || print_error "Namespace not found"
    echo ""
    check_running_engine

    echo -e "${YELLOW}═══ Pod Management ═══${NC}"
    echo "  1) View all pods"
    echo "  2) View all services"
    echo "  3) View all deployments"
    echo ""

    echo -e "${YELLOW}═══ Inference Engine Switching ═══${NC}"
    echo "  4) Switch to Transformers"
    echo "  5) Switch to vLLM"
    echo "  6) Switch to TensorRT"
    echo "  7) Stop all inference engines"
    echo ""

    echo -e "${YELLOW}═══ Logs ═══${NC}"
    echo "  8) View Orchestrator logs"
    echo "  9) View Transformers logs"
    echo " 10) View vLLM logs"
    echo " 11) View TensorRT logs"
    echo " 12) Follow Orchestrator logs (live)"
    echo " 13) Follow active inference engine logs (live)"
    echo ""

    echo -e "${YELLOW}═══ Port Forwarding ═══${NC}"
    echo " 14) Port-forward Orchestrator (8011)"
    echo " 15) Port-forward Transformers (8012)"
    echo " 16) Port-forward vLLM (8013)"
    echo " 17) Port-forward TensorRT (8014)"
    echo ""

    echo -e "${YELLOW}═══ Health & Diagnostics ═══${NC}"
    echo " 18) Check node status"
    echo " 19) Check GPU allocation"
    echo " 20) Describe orchestrator pod"
    echo " 21) Describe active inference pod"
    echo " 22) Get events in namespace"
    echo ""

    echo -e "${YELLOW}═══ Quick Actions ═══${NC}"
    echo " 23) Restart orchestrator"
    echo " 24) Restart active inference engine"
    echo " 25) Quick health check (all services)"
    echo ""

    echo -e "${YELLOW}═══ System ═══${NC}"
    echo " 26) Remove control-plane taint"
    echo " 27) Clean up failed/unknown pods"
    echo ""

    echo -e "${RED}  0) Exit${NC}"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice: " choice

    case $choice in
        1)
            print_header "All Pods"
            kubectl get pods -n $NAMESPACE
            read -p "Press Enter to continue..."
            ;;
        2)
            print_header "All Services"
            kubectl get svc -n $NAMESPACE
            read -p "Press Enter to continue..."
            ;;
        3)
            print_header "All Deployments"
            kubectl get deployments -n $NAMESPACE
            read -p "Press Enter to continue..."
            ;;
        4)
            switch_engine "transformers"
            read -p "Press Enter to continue..."
            ;;
        5)
            switch_engine "vllm"
            read -p "Press Enter to continue..."
            ;;
        6)
            switch_engine "tensorrt"
            read -p "Press Enter to continue..."
            ;;
        7)
            print_header "Stopping all inference engines"
            kubectl scale deployment llm-transformers -n $NAMESPACE --replicas=0
            kubectl scale deployment llm-vllm -n $NAMESPACE --replicas=0
            kubectl scale deployment llm-tensorrt -n $NAMESPACE --replicas=0
            sleep 3
            kubectl get pods -n $NAMESPACE
            print_success "All inference engines stopped"
            read -p "Press Enter to continue..."
            ;;
        8)
            view_logs "llm-orchestrator" "static"
            read -p "Press Enter to continue..."
            ;;
        9)
            view_logs "llm-transformers" "static"
            read -p "Press Enter to continue..."
            ;;
        10)
            view_logs "llm-vllm" "static"
            read -p "Press Enter to continue..."
            ;;
        11)
            view_logs "llm-tensorrt" "static"
            read -p "Press Enter to continue..."
            ;;
        12)
            view_logs "llm-orchestrator" "follow"
            ;;
        13)
            # Find which engine is running
            if [ "$TRANSFORMERS_REPLICAS" != "0" ]; then
                view_logs "llm-transformers" "follow"
            elif [ "$VLLM_REPLICAS" != "0" ]; then
                view_logs "llm-vllm" "follow"
            elif [ "$TENSORRT_REPLICAS" != "0" ]; then
                view_logs "llm-tensorrt" "follow"
            else
                print_error "No inference engine is running"
                read -p "Press Enter to continue..."
            fi
            ;;
        14)
            port_forward "llm-orchestrator" "8011"
            ;;
        15)
            port_forward "llm-transformers" "8012"
            ;;
        16)
            port_forward "llm-vllm" "8013"
            ;;
        17)
            port_forward "llm-tensorrt" "8014"
            ;;
        18)
            print_header "Node Status"
            kubectl get nodes -o wide
            read -p "Press Enter to continue..."
            ;;
        19)
            print_header "GPU Allocation"
            kubectl describe nodes | grep -A 10 "Allocated resources"
            read -p "Press Enter to continue..."
            ;;
        20)
            print_header "Orchestrator Pod Details"
            kubectl describe pod -l app=llm-orchestrator -n $NAMESPACE
            read -p "Press Enter to continue..."
            ;;
        21)
            print_header "Active Inference Pod Details"
            if [ "$TRANSFORMERS_REPLICAS" != "0" ]; then
                kubectl describe pod -l app=llm-transformers -n $NAMESPACE
            elif [ "$VLLM_REPLICAS" != "0" ]; then
                kubectl describe pod -l app=llm-vllm -n $NAMESPACE
            elif [ "$TENSORRT_REPLICAS" != "0" ]; then
                kubectl describe pod -l app=llm-tensorrt -n $NAMESPACE
            else
                print_error "No inference engine is running"
            fi
            read -p "Press Enter to continue..."
            ;;
        22)
            print_header "Recent Events"
            kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'
            read -p "Press Enter to continue..."
            ;;
        23)
            print_header "Restarting Orchestrator"
            kubectl rollout restart deployment llm-orchestrator -n $NAMESPACE
            sleep 3
            kubectl get pods -n $NAMESPACE -l app=llm-orchestrator
            print_success "Orchestrator restarted"
            read -p "Press Enter to continue..."
            ;;
        24)
            print_header "Restarting Active Inference Engine"
            if [ "$TRANSFORMERS_REPLICAS" != "0" ]; then
                kubectl rollout restart deployment llm-transformers -n $NAMESPACE
                print_success "Transformers restarted"
            elif [ "$VLLM_REPLICAS" != "0" ]; then
                kubectl rollout restart deployment llm-vllm -n $NAMESPACE
                print_success "vLLM restarted"
            elif [ "$TENSORRT_REPLICAS" != "0" ]; then
                kubectl rollout restart deployment llm-tensorrt -n $NAMESPACE
                print_success "TensorRT restarted"
            else
                print_error "No inference engine is running"
            fi
            sleep 3
            kubectl get pods -n $NAMESPACE
            read -p "Press Enter to continue..."
            ;;
        25)
            print_header "Quick Health Check"
            echo "Orchestrator:"
            kubectl get pods -l app=llm-orchestrator -n $NAMESPACE
            echo ""
            echo "Inference Engines:"
            kubectl get pods -l component=inference-engine -n $NAMESPACE
            echo ""
            echo "Services:"
            kubectl get svc -n $NAMESPACE
            read -p "Press Enter to continue..."
            ;;
        26)
            print_header "Removing Control-Plane Taint"
            kubectl taint nodes --all node-role.kubernetes.io/control-plane- 2>/dev/null && print_success "Taint removed" || print_warning "Taint not found (already removed)"
            read -p "Press Enter to continue..."
            ;;
        27)
            print_header "Cleaning Up Failed/Unknown Pods"
            kubectl delete pod -n $NAMESPACE --field-selector=status.phase!=Running,status.phase!=Pending --force --grace-period=0 2>/dev/null && print_success "Cleanup complete" || print_warning "No pods to clean up"
            read -p "Press Enter to continue..."
            ;;
        0)
            print_success "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            sleep 2
            ;;
    esac
done
