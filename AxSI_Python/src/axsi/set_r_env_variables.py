import os
import socket
import subprocess
import sys


def find_r_home():
    """Find R_HOME dynamically using the R executable."""
    try:
        # Run R command to get R_HOME
        result = subprocess.run(["R", "RHOME"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        r_home = result.stdout.strip()
        return r_home
    except FileNotFoundError:
        print("Error: R is not installed or not in the PATH.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError:
        print("Error: Unable to determine R_HOME. Check your R installation.", file=sys.stderr)
        return None


def set_r_environment():
    """Set R_HOME and update PATH based on the detected R installation."""
    r_home = find_r_home()
    if not r_home:
        print("R is not installed or configured properly. Please install R from https://cran.r-project.org/",
              file=sys.stderr)
        sys.exit(1)

    # Set R_HOME environment variable
    os.environ['R_HOME'] = r_home
    print(f"R_HOME set to: {r_home}")

    # Update PATH to include R bin directory
    r_bin_path = os.path.join(r_home, "bin")
    if r_bin_path not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + r_bin_path
        print(f"R bin directory added to PATH: {r_bin_path}")

    # Verify R is now accessible
    try:
        subprocess.run(["R", "--version"], check=True)
        print("R is properly configured and accessible.")
    except Exception as e:
        print("Error: R is still not accessible. Ensure it is properly installed.", file=sys.stderr)
        sys.exit(1)


def check_port_availability(port):
    """Check if a port is available (not occupied)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True  # Port is available
        except socket.error:
            return False  # Port is occupied


# Not in use - R server is very slow
def upload_rserver(port=6133):
    if not check_port_availability(port):
        return False
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', port))
        server_socket.listen(5)  # Start listening for connections

        print(f"Server is listening on port {port}...")
        return server_socket
    except Exception as e:
        print(f"Error: {e}")


# Call this function during setup or at runtime as needed
if __name__ == "__main__":
    set_r_environment()
    upload_rserver(int(sys.argv[1]))  # port number (int) - the default is 6133
