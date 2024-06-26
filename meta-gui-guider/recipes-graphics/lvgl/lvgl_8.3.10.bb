# SPDX-License-Identifier: MIT

HOMEPAGE = "https://lvgl.io/"
DESCRIPTION = "LVGL is an OSS graphics library to create embedded GUI"
SUMMARY = "Light and Versatile Graphics Library"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENCE;md5=bf1198c89ae87f043108cea62460b03a"

include lvgl.inc

SRC_URI = "file://lvgl.zip;subdir=${S} \
    file://lv_conf.h \
    file://lv_conf_ext.h \
"

REQUIRED_DISTRO_FEATURES = "wayland"

inherit cmake
inherit features_check

EXTRA_OECMAKE = "-DLIB_INSTALL_DIR=${BASELIB} -DLV_CONF_BUILD_DISABLE_EXAMPLES=1 -DLV_CONF_BUILD_DISABLE_DEMOS=1"

do_configure:prepend() {
    cp "${WORKDIR}/lv_conf.h" ${S}
    cp "${WORKDIR}/lv_conf_ext.h" ${S}
}

do_install:append() {
    install -m 0644 ${S}/lv_conf_ext.h ${D}${includedir}/${PN}/
}

FILES:${PN}-dev += "\
    ${includedir}/${PN}/ \
    ${includedir}/${PN}/lvgl/ \
    "
