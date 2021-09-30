#pragma once
#include "InfrastAbstractTask.h"

namespace asst {
    // 截图任务，主要是调试用的，制作模板匹配的素材啥的
    class ScreenCaptureTask : public InfrastAbstractTask
    {
    public:
        using InfrastAbstractTask::InfrastAbstractTask;
        virtual ~ScreenCaptureTask() = default;

        virtual bool run() override;
    protected:
        virtual bool cap_opers_name_in_list();
        virtual bool cap_opers_name_in_infrast();
    };
}