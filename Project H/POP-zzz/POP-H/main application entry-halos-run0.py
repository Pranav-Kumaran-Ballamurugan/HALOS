def main():
    # Initialize core system
    core = HALOSCore()
    
    # Start GUI or server based on mode
    if os.getenv('HALOS_MODE') == 'server':
        server = HALOSServer(core)
        server.run()
    else:
        gui = HALOSGUI(core)
        gui.window.show()
        sys.exit(gui.app.exec_())

if __name__ == "__main__":
    main()