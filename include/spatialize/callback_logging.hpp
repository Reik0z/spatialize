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
            CallbackLogger(std::function<int(std::string)> callback, std::string caller_classname = ""): CallbackSender(callback) {
                this->caller_classname = caller_classname;
            }

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
            std::string caller_classname;
            std::string build_message(std::string msg, std::string level) {
                this->json.str("");
				this->json << "{\"message\": { \"text\":\"" << "[C++|" << this->caller_classname << "] " << msg << "\", \"level\":\"" << level <<"\"}}";
				return this->json.str();
            }
    };

    class CallbackProgressSender: public sptlz::CallbackSender {
        public:
            CallbackProgressSender(std::function<int(std::string)> callback): CallbackSender(callback) {}

            ~CallbackProgressSender(){}

            void init(int total, int increment_step) {
				// {"progress": {"init": <total expected count>, "step": <increment step>}}
				this->json.str("");
				this->json << "{\"progress\": { \"init\":" << total << ", \"step\":" << increment_step <<"}}";
				this->callback(this->json.str());
            }

            void inform(int value) {
            	// {"progress": {"token": <value>}}
				this->json.str("");
				this->json << "{\"progress\": {\"token\":" << value << "}}";
				this->callback(this->json.str());
            }

            void stop (){
                // {"progress": "done"}
				this->json.str("");
				this->json << "{\"progress\": \"done\"}";
				this->callback(this->json.str());
            }
    };
}

#endif


