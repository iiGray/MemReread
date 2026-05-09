import os, sys, time, signal, subprocess
from typing import Optional

def calculate_tp_size(gpus: str) -> int:
    return len([g for g in gpus.split(',') if g.strip()])


class VllmWorker:
    def __init__(self, model_path: str, port: int, gpus: str = "0,1,2,3,4,5,6,7", parser: str = "", max_model_len: int = 40960, seqs: int = 256,
    log_dir: str = './vllm_logs', wait_for_ready = True, timeout = 120):
        self.process = None
        self.port = port

        self.start_vllm_server(
            model_path, port, gpus, parser, max_model_len, seqs, log_dir, wait_for_ready, timeout
        )


    def start_vllm_server(
        self, 
        model_path: str,
        port: int,
        gpus: str,
        parser: str = "",
        max_model_len: int = 40960,
        seqs: int = 256,
        log_dir: str = "./vllm_logs",
        wait_for_ready: bool = True,
        timeout: int = 120
    ) -> subprocess.Popen:

        if self.process is not None and self.process.poll() is None:
            print(f"[WARNING]  (PID: {self.process.pid}) is already running !!!")
            return self.process

        tp_size = calculate_tp_size(gpus)
        print(f"[INFO] Tensor Parallel Size: {tp_size}")
        print(f"[INFO] Using GPUs: {gpus}")

        model_path = model_path.rstrip('/')
        self.port = port 

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpus
        env['VLLM_USE_V1'] = '0'
        env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"vllm_server_port{port}.log")

        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--served-model-name", model_path,
            "--trust-remote-code",
            "--dtype", "float16",
            "--port", str(port),
            "--tensor-parallel-size", str(tp_size),
            "--gpu-memory-utilization", "0.98",
            "--disable-custom-all-reduce",
            "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(seqs),
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "32768",
            "--disable-log-requests",
        ]
        if "Qwen2.5" in model_path:
            cmd += ["--enforce-eager"]
        
        print(f"[INFO] Start Command: {' '.join(cmd)}")
        print(f"[INFO] Log file: {log_file}")
        
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        
        print(f"[INFO] vLLM has been started, PID: {process.pid}")
        
        if wait_for_ready:
            if self.wait_for_server_ready(port, timeout):
                print(f"[INFO] has been ready: http://localhost:{port}")
            else:
                print(f"[WARNING] Timeout ...")
        
        self.process = process
        return process

    def wait_for_server_ready(self, port: int, timeout: int = 120) -> bool:

        import socket
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result == 0:
                    time.sleep(2)
                    return True
            except Exception:
                pass
            time.sleep(2)
            print(f"[INFO] Waiting Server... ({int(time.time() - start_time)}/{timeout}s)")
        return False

    def stop(self) -> bool: 

        process = self.process
        
        if process is None:
            return True
            
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                print(f"[INFO] Send Terminal Signal to {process.pid}")
                try:
                    process.wait(timeout=10)
                    print(f"[INFO] Exit successfully.")
                    self.process = None
                    return True
                except subprocess.TimeoutExpired:
                    print(f"[WARNING] Exit Failed, Force to terminate.")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    self.process = None
                    return True
            except Exception as e:
                print(f"[ERROR] Terminate Failed: {e}")
                return False
        else:
            print(f"[INFO] Exit, code: {process.returncode}")
            self.process = None
            return True

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            import traceback
            print("\n" + "="*50)
            traceback.print_exception(exc_type, exc_val, exc_tb)
            print("="*50 + "\n")
        self.stop()
        return False