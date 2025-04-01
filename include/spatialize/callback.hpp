#ifndef _SPTLZ_CB_
#define _SPTLZ_CB_

#include <functional>

namespace sptlz{

    class CallbackSender {
        protected:
            std::stringstream json;
            std::function<int(std::string)> callback;
        public:
            CallbackSender(std::function<int(std::string)> callback) {
                this->callback = callback;
            }
            ~CallbackSender(){
            }
    };
}

#endif