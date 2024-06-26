# SPDX-License-Identifier: MIT

HOMEPAGE = "https://www.nxp.com/design/software/development-software/gui-guider:GUI-GUIDER"
DESCRIPTION = "A user-friendly graphical user interface development tool"
LICENSE = "Proprietary"
LIC_FILES_CHKSUM = "file://LICENSE.txt;md5=9c7719a39d365a6b1dcd6b001243ff28"

SRC_URI = "file://gui-guider.zip;subdir=${S}"
include gui-guider.inc

REQUIRED_DISTRO_FEATURES = "wayland"

DEPENDS = "libxkbcommon lvgl wayland lv-drivers"

inherit cmake
inherit features_check
inherit pkgconfig

TARGET_CFLAGS += "-DLV_WAYLAND_TIMER_HANDLER"
TARGET_CFLAGS += "-I${RECIPE_SYSROOT}/${includedir}/lvgl"
TARGET_CFLAGS += "-I${RECIPE_SYSROOT}/${includedir}/lvgl/src"
TARGET_CFLAGS += "-I${RECIPE_SYSROOT}/${includedir}/lvgl/lv_drivers"

