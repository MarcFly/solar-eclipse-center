"""
Root entrypoint for the Solar Eclipse Bulk Orchestrator.
Delegates to eclipse_processor.cli.main().
"""
from eclipse_processor.cli import main

if __name__ == "__main__":
    main()
