/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <nvcuvid.h>


#include <windows.h>
#define sleep(x) Sleep(x)


class FrameQueue
{
    public:
        static const unsigned int cnMaximumSize = 40; // MAX_FRM_CNT;

        FrameQueue();

        virtual
        ~FrameQueue();

        void
        waitForQueueUpdate();

        void
        enter_CS(CRITICAL_SECTION *pCS);

        void
        leave_CS(CRITICAL_SECTION *pCS);

        void
        set_event(HANDLE event);

        void
        reset_event(HANDLE event);

        void
        enqueue(const CUVIDPARSERDISPINFO *pPicParams);

        // Deque the next frame.
        // Parameters:
        //      pDisplayInfo - New frame info gets placed into this object.
        //          Note: This pointer must point to a valid struct. The method
        //          does not create memory for this.
        // Returns:
        //      true, if a new frame was returned,
        //      false, if the queue was empty and no new frame could be returned.
        //          In that case, pPicParams doesn't contain valid data.
        bool
        dequeue(CUVIDPARSERDISPINFO *pDisplayInfo);

        void
        releaseFrame(const CUVIDPARSERDISPINFO *pPicParams);

        bool
        isInUse(int nPictureIndex)
        const;

        bool
        isEndOfDecode()
        const;

        void
        endDecode();

        // Spins until frame becomes available or decoding
        // gets canceled.
        // If the requested frame is available the method returns true.
        // If decoding was interupted before the requested frame becomes
        // available, the method returns false.
        bool
        waitUntilFrameAvailable(int nPictureIndex);

    private:
        void
        signalStatusChange();

        HANDLE hEvent_;
        CRITICAL_SECTION    oCriticalSection_;
        volatile int        nReadPosition_;
        volatile int        nFramesInQueue_;
        CUVIDPARSERDISPINFO aDisplayQueue_[cnMaximumSize];
        volatile int        aIsFrameInUse_[cnMaximumSize];
        volatile int        bEndOfDecode_;
};

