# Copyright (c) OpenMMLab. All rights reserved.
# Simple script to disable ASLR and make .nv_fatb sections read-only
# Requires: pefile  ( python -m pip install pefile )
# Usage:  fixNvPe.py --input path/to/*.dll
# yapf: disable
import argparse
import glob
import os
import shutil
import subprocess

import pefile
import torch


def main(inputs, args):
    if not torch.cuda.is_available():
        return
    failures = []
    for file in glob.glob(inputs, recursive=args.recursive):
        print(f'\n---\nChecking {file}...')
        pe = pefile.PE(file, fast_load=True)
        nvbSect = [
            section for section in pe.sections
            if section.Name.decode().startswith('.nv_fatb')
        ]
        if len(nvbSect) == 1:
            sect = nvbSect[0]
            size = sect.Misc_VirtualSize
            aslr = pe.OPTIONAL_HEADER.IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
            writable = 0 != (
                sect.Characteristics
                & pefile.SECTION_CHARACTERISTICS['IMAGE_SCN_MEM_WRITE'])
            print(f'Found NV FatBin! Size: {size/1024/1024:0.2f}MB  ASLR: '
                  f'{aslr}  Writable: {writable}')
            if (writable or aslr) and size > 0:
                print('- Modifying DLL')
                if args.backup:
                    bakFile = f'{file}_bak'
                    print(f'- Backing up [{file}] -> [{bakFile}]')
                    if os.path.exists(bakFile):
                        print(f'- Warning: Backup file already exists '
                              f'({bakFile}), not modifying file! Delete the '
                              f"'bak' to allow modification")
                        failures.append(file)
                        continue
                    try:
                        shutil.copy2(file, bakFile)
                    except Exception as e:
                        print(f'- Failed to create backup! [{str(e)}], '
                              f'not modifying file!')
                        failures.append(file)
                        continue
                # Disable ASLR for DLL, and disable writing for section
                pe.OPTIONAL_HEADER.DllCharacteristics &= ~pefile.DLL_CHARACTERISTICS['IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE']  # noqa: E501
                sect.Characteristics = sect.Characteristics & ~pefile.SECTION_CHARACTERISTICS['IMAGE_SCN_MEM_WRITE']  # noqa: E501
                try:
                    newFile = f'{file}_mod'
                    print(f'- Writing modified DLL to [{newFile}]')
                    pe.write(newFile)
                    pe.close()
                    print(f'- Moving modified DLL to [{file}]')
                    os.remove(file)
                    shutil.move(newFile, file)
                except Exception as e:
                    print(f'- Failed to write modified DLL! [{str(e)}]')
                    failures.append(file)
                    continue

    print('\n\nDone!')
    if len(failures) > 0:
        print('***WARNING**** These files needed modification but failed: ')
        for failure in failures:
            print(f' - {failure}')


def parseArgs():
    parser = argparse.ArgumentParser(
        description='Disable ASLR and make .nv_fatb sections read-only',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--backup', help='Backup modified files', default=True, required=False)
    parser.add_argument(
        '--recursive',
        '-r',
        default=False,
        action='store_true',
        help='Recurse into subdirectories')

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == '__main__':
    out = subprocess.check_output(['pip', 'show', 'torch'])
    location = out.decode('utf-8').split('\n')
    for line in location:
        if line.startswith('Location: '):
            site_package_path = line.replace('Location: ', '')
            site_package_path.rstrip('\r')
    torch_path = f'{site_package_path}/torch/lib/*.dll'
    print(torch_path)
    tmp_path = f'{site_package_path}/torch/lib'
    print(f'torch_path exists: {os.path.exists(tmp_path)}')
    args = parseArgs()
    main(torch_path, args)
