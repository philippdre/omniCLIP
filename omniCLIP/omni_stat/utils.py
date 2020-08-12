"""
    omniCLIP is a CLIP-Seq peak caller

    Copyright (C) 2017 Philipp Boss

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import resource
import time


def get_mem_usage(verbosity, t=False, msg=''):
    """Return the maximum resident set size with optional message."""
    if verbosity:
        if msg:
            print(msg)
        if t:
            print('Done: Elapsed time: ' + str(time.time() - t))

        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print('Memory usage: %s (kb)' % mem_usage)
