/*
 * Magicnumbers.h
 *
 *  Created on: 06.08.2013
 *      Author: christoph
 */

#ifndef MAGICNUMBERS_H_
#define MAGICNUMBERS_H_


const UINT8 KEY_PRESSED = 1;
const UINT8 KEY_RELEASED = 2;
const UINT8 SPECIAL_KEY_PRESSED = 3;
const UINT8 SPECIAL_KEY_RELEASED = 4;
const UINT8 SHUTDOWN_CONNECTION = 5;
const UINT8 WINDOW_SIZE = 7;
const UINT8 MOUSE_PRESSED = 8;
const UINT8 MOUSE_RELEASED = 9;

//Identifyer to send Data over TCP connection: UINT8 FRAME_DATA; INT SIZE; SIZE * sizeof(UINT8) char;
const UINT8 FRAME_DATA = 6;

#endif