/*************************************************************************

Copyright 2014 Christoph Eichler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*************************************************************************/

#pragma once

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
const UINT8 FRAME_DATA_MEASURE = 10;

//GFXAPI Settings
const UINT8 GFX_GL = 11;
const UINT8 GFX_D3D = 12;

//Message Broadcasting
const UINT8 BROADCAST_MESSAGE = 13;
