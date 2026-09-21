import threading
import time
import webbrowser

import server as srv


def main():
    srv.load_or_train_agent()
    httpd = srv.ThreadingHTTPServer(("localhost", srv.PORT), srv.Handler)
    url = f"http://localhost:{srv.PORT}/cube_visualizer_local.html"

    def open_browser():
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\nServing on {url}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
