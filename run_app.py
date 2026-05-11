import webbrowser
import threading
import time
from web_app import app

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    print("🚀 Démarrage de l'application...")
    print("📁 Le navigateur va s'ouvrir automatiquement")
    print("⚠️  Pour arrêter : fermez cette fenêtre ou faites Ctrl+C")
    app.run(debug=False, host='127.0.0.1', port=5000)
    