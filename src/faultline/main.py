"""FAULTLINE CLI entry point."""

import sys


def main() -> None:
    company = sys.argv[1] if len(sys.argv) > 1 else None
    if not company:
        print("Usage: python -m faultline.main <COMPANY>")
        sys.exit(1)
    print(f"FAULTLINE pipeline initialized for: {company}")
    print("Pipeline stages will execute once tasks are implemented.")


if __name__ == "__main__":
    main()
