#ifndef _SPTLZ_LOGGING_
#define _SPTLZ_LOGGING_

#include <functional>
#include <string>

namespace sptlz{
    namespace logger{
        const struct {
            std::string debug = "DEBUG";
            std::string info = "INFO";
            std::string warn = "WARNING";
            std::string error = "ERROR";
            std::string critical = "CRITICAL";
        } level;
    }

    class Logger {
        protected:
            virtual void debug(std::string msg){
                throw std::runtime_error("must override");
            }

            virtual void info(std::string msg){
                throw std::runtime_error("must override");
            }

            virtual void warning(std::string msg){
                throw std::runtime_error("must override");
            }

            virtual void error(std::string msg){
                throw std::runtime_error("must override");
            }

            virtual void critical(std::string msg){
                throw std::runtime_error("must override");
            }
        public:
            virtual ~Logger();
    };

    class CallbackLogger: public Logger {
        /* Send logging to Python
            PROTOCOL
            {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
        */
        private:
            std::stringstream json;
            std::function<int(std::string)> callback;
        public:
            CallbackLogger(std::function<int(std::string)> callback) {
                this->callback = callback;
            }

            ~CallbackLogger(){
            }

            void debug(std::string msg){
                json.str("");
				json << "{\"message\": { \"text\":\"" << msg << "\", \"level\":\"" << "DEBUG" <<"\"}}";
				callback(json.str());
            }
    };
}

#endif


