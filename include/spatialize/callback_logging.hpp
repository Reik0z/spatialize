#ifndef _SPTLZ_CB_LOGGING_
#define _SPTLZ_CB_LOGGING_

#include <functional>
#include <string>
#include "callback.hpp"

namespace sptlz{

    const struct {
        const struct {
            std::string debug = "DEBUG";
            std::string info = "INFO";
            std::string warn = "WARNING";
            std::string error = "ERROR";
            std::string critical = "CRITICAL";
        } level;
    } logger;

    class CallbackLogger: public sptlz::CallbackSender {
        /* Send logging to Python
            PROTOCOL
            {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
        */
        public:
            CallbackLogger(std::function<int(std::string)> callback): CallbackSender(callback) {}

            ~CallbackLogger(){}

            void debug(std::string msg){
                this->callback(build_message(msg, logger.level.debug));
            }

            void info(std::string msg){
				this->callback(build_message(msg, logger.level.info));
            }

            void warning(std::string msg){
				this->callback(build_message(msg, logger.level.warn));
            }

            void error(std::string msg){
				this->callback(build_message(msg, logger.level.error));
            }

            void critical(std::string msg){
				this->callback(build_message(msg, logger.level.critical));
            }
        private:
            std::string build_message(std::string msg, std::string level) {
                this->json.str("");
				this->json << "{\"message\": { \"text\":\"" << msg << "\", \"level\":\"" << level <<"\"}}";
				return json.str();
            }
    };
}

#endif


