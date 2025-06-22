#!/usr/bin/env python3
"""Script principal para ejecutar el sistema"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    main()
