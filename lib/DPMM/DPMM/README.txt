This is a matlab library for Gaussian Dirichlet Process Mixture Models (DPMMs).
It includes both variational and Monte Carlo inference.

To test / see how this program works, run demodpmm.m in matlab

This code was mostly written in 2007. When I found out it was
referenced in a paper in 2012, I made a few cosmetic changes and put
it on Github. It's not guaranteed to work perfectly. You should
check it before using it for anything really important.

Some of the sampling code is built on top of software by Michael
Mandel, which was also released under GPL.

=====================================================
COPYRIGHT / LICENSE
=====================================================
Unless otherwise indicated in the specific file, code was written by 
Jacob Eisenstein, and is copyrighted under the GPL:

  Copyright (C) 2007-2012  Jacob Eisenstein

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.



JAHANDAR:
1) this file is using older versions of creating mex files with "mexopts.bat". But there is no such a file in the newer versions.
create mexopts.bat in "%USER_ENV%\AppData\Roaming\Mathworks\MATLAB\R2016a"
copy the following lines:
"""
@echo off
rem MSVC110OPTS.BAT
rem
rem    Compile and link options used for building MEX-files
rem    using the Microsoft Visual C++ compiler version 11.0
rem
rem    $Revision: 1.1.6.1 $  $Date: 2012/09/25 18:20:03 $
rem    Copyright 2007-2012 The MathWorks, Inc.
rem
rem StorageVersion: 1.0
rem C++keyFileName: MSVC110OPTS.BAT
rem C++keyName: Microsoft Visual C++ 2012
rem C++keyManufacturer: Microsoft
rem C++keyVersion: 11.0
rem C++keyLanguage: C++
rem C++keyLinkerName: Microsoft Visual C++ 2012
rem C++keyLinkerVer: 11.0
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************

set MATLAB=%MATLAB%
set VSINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 11.0
set VCINSTALLDIR=%VSINSTALLDIR%\VC
rem In this case, LINKERDIR is being used to specify the location of the SDK
set LINKERDIR=C:\Program Files (x86)\Windows Kits\8.0\
set PATH=%VCINSTALLDIR%\bin;%VCINSTALLDIR%\VCPackages;%VSINSTALLDIR%\Common7\IDE;%VSINSTALLDIR%\Common7\Tools;%LINKERDIR%\bin\x86;%LINKERDIR%\bin;%MATLAB_BIN%;%PATH%
set INCLUDE=%VCINSTALLDIR%\INCLUDE;%VCINSTALLDIR%\ATLMFC\INCLUDE;%LINKERDIR%\include\um;%LINKERDIR%\include\shared;%LINKERDIR%\include\winrt;%INCLUDE%
set LIB=%VCINSTALLDIR%\LIB;%VCINSTALLDIR%\ATLMFC\LIB;%LINKERDIR%\lib\win8\um\x86;%MATLAB%\extern\lib\win32;%LIB%
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=cl
set COMPFLAGS=/c /GR /W3 /EHs /D_CRT_SECURE_NO_DEPRECATE /D_SCL_SECURE_NO_DEPRECATE /D_SECURE_SCL=0 /DMATLAB_MEX_FILE /nologo /MD
set OPTIMFLAGS=/O2 /Oy- /DNDEBUG /openmp
set DEBUGFLAGS=/Z7
set NAME_OBJECT=/Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win64\microsoft
set LINKER=link
set LINKFLAGS=/dll /export:%ENTRYPOINT% /LIBPATH:"%LIBLOC%" libmx.lib libmex.lib libmat.lib /MACHINE:X86 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /manifest /incremental:NO /implib:"%LIB_NAME%.x" /MAP:"%OUTDIR%%MEX_NAME%%MEX_EXT%.map"
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug /PDB:"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb"
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%mexversion.res"
set RC_LINKER=

set POSTLINK_CMDS=del "%LIB_NAME%.x" "%LIB_NAME%.exp"
set POSTLINK_CMDS1=mt -outputresource:"%OUTDIR%%MEX_NAME%%MEX_EXT%;2" -manifest "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
set POSTLINK_CMDS2=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
set POSTLINK_CMDS3=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.map"
"""



2) if in the lightspeed folder you have util.obj , in install_fastfit.m change "util.o" to "util.obj"