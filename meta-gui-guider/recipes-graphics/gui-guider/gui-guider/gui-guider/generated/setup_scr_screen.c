/*
* Copyright 2024 NXP
* NXP Confidential and Proprietary. This software is owned or controlled by NXP and may only be used strictly in
* accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing,
* activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to
* comply with and are bound by, such license terms.  If you do not agree to be bound by the applicable license
* terms, then you may not retain, install, activate or otherwise use the software.
*/

#include "lvgl.h"
#include <stdio.h>
#include "gui_guider.h"
#include "events_init.h"
#include "widgets_init.h"
#include "custom.h"



void setup_scr_screen(lv_ui *ui)
{
	//Write codes screen
	ui->screen = lv_obj_create(NULL);
	lv_obj_set_size(ui->screen, 1280, 800);
	lv_obj_set_scrollbar_mode(ui->screen, LV_SCROLLBAR_MODE_OFF);

	//Write style for screen, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_bg_opa(ui->screen, 255, LV_PART_MAIN|LV_STATE_DEFAULT);
	lv_obj_set_style_bg_color(ui->screen, lv_color_hex(0xcbf2ff), LV_PART_MAIN|LV_STATE_DEFAULT);
	lv_obj_set_style_bg_grad_dir(ui->screen, LV_GRAD_DIR_NONE, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_1
	ui->screen_img_1 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_1, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_pivot(ui->screen_img_1, 50,50);
	lv_img_set_angle(ui->screen_img_1, 0);
	lv_obj_set_pos(ui->screen_img_1, 642, 115);
	lv_obj_set_size(ui->screen_img_1, 570, 439);

	//Write style for screen_img_1, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_1, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_qrcode_1
	ui->screen_qrcode_1 = lv_qrcode_create(ui->screen, 132, lv_color_hex(0x2C3224), lv_color_hex(0xffffff));
	const char * screen_qrcode_1_data = "github.com/RichardGajdosik/Gesture_Recognition_NXP_VUTFIT";
	lv_qrcode_update(ui->screen_qrcode_1, screen_qrcode_1_data, strlen(screen_qrcode_1_data));
	lv_obj_set_pos(ui->screen_qrcode_1, 41, 625);
	lv_obj_set_size(ui->screen_qrcode_1, 132, 132);

	//Write codes screen_img_2
	ui->screen_img_2 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_2, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_src(ui->screen_img_2, &_dislike_alpha_100x100);
	lv_img_set_pivot(ui->screen_img_2, 50,50);
	lv_img_set_angle(ui->screen_img_2, 0);
	lv_obj_set_pos(ui->screen_img_2, 693, 633);
	lv_obj_set_size(ui->screen_img_2, 100, 100);

	//Write style for screen_img_2, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_2, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_3
	ui->screen_img_3 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_3, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_src(ui->screen_img_3, &_fist_alpha_100x100);
	lv_img_set_pivot(ui->screen_img_3, 50,50);
	lv_img_set_angle(ui->screen_img_3, 0);
	lv_obj_set_pos(ui->screen_img_3, 793, 633);
	lv_obj_set_size(ui->screen_img_3, 100, 100);

	//Write style for screen_img_3, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_3, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_4
	ui->screen_img_4 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_4, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_src(ui->screen_img_4, &_like_alpha_100x100);
	lv_img_set_pivot(ui->screen_img_4, 50,50);
	lv_img_set_angle(ui->screen_img_4, 0);
	lv_obj_set_pos(ui->screen_img_4, 893, 633);
	lv_obj_set_size(ui->screen_img_4, 100, 100);

	//Write style for screen_img_4, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_4, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_5
	ui->screen_img_5 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_5, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_src(ui->screen_img_5, &_peace_alpha_100x100);
	lv_img_set_pivot(ui->screen_img_5, 50,50);
	lv_img_set_angle(ui->screen_img_5, 0);
	lv_obj_set_pos(ui->screen_img_5, 993, 633);
	lv_obj_set_size(ui->screen_img_5, 100, 100);

	//Write style for screen_img_5, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_5, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_6
	ui->screen_img_6 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_6, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_src(ui->screen_img_6, &_stop_alpha_100x100);
	lv_img_set_pivot(ui->screen_img_6, 50,50);
	lv_img_set_angle(ui->screen_img_6, 0);
	lv_obj_set_pos(ui->screen_img_6, 1093, 633);
	lv_obj_set_size(ui->screen_img_6, 100, 100);

	//Write style for screen_img_6, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_6, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_7
	ui->screen_img_7 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_7, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_pivot(ui->screen_img_7, 50,50);
	lv_img_set_angle(ui->screen_img_7, 0);
	lv_obj_set_pos(ui->screen_img_7, 208, 324);
	lv_obj_set_size(ui->screen_img_7, 234, 206);

	//Write style for screen_img_7, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_7, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_8
	ui->screen_img_8 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_8, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_src(ui->screen_img_8, &_vut_fit_logo_alpha_337x138);
	lv_img_set_pivot(ui->screen_img_8, 50,50);
	lv_img_set_angle(ui->screen_img_8, 0);
	lv_obj_set_pos(ui->screen_img_8, 3, 4);
	lv_obj_set_size(ui->screen_img_8, 337, 138);

	//Write style for screen_img_8, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_8, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//Write codes screen_img_9
	ui->screen_img_9 = lv_img_create(ui->screen);
	lv_obj_add_flag(ui->screen_img_9, LV_OBJ_FLAG_CLICKABLE);
	lv_img_set_src(ui->screen_img_9, &_nxp_alpha_295x135);
	lv_img_set_pivot(ui->screen_img_9, 50,50);
	lv_img_set_angle(ui->screen_img_9, 0);
	lv_obj_set_pos(ui->screen_img_9, 982, 4);
	lv_obj_set_size(ui->screen_img_9, 295, 135);

	//Write style for screen_img_9, Part: LV_PART_MAIN, State: LV_STATE_DEFAULT.
	lv_obj_set_style_img_opa(ui->screen_img_9, 255, LV_PART_MAIN|LV_STATE_DEFAULT);

	//The custom code of screen.
	

	//Update current screen layout.
	lv_obj_update_layout(ui->screen);

}
