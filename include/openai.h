#pragma once

#include <iostream>
#include <fstream>
#include <tuple>
#include <stdexcept>
#include <nlohmann/json.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

using namespace std;
using namespace httplib;

using json = nlohmann::json;

namespace openai {
    using session_result = tuple<int, string> ;

    class Session {
        Client cli_;

        public:
            Session(const string& scheme_host_port, bool verbose = false);

            void stop();

            void set_token(const string& token);
            void set_proxy(const string& host, int port);

            session_result get(const string& path);
            session_result post(const string& path, 
                                const string& data, 
                                const string& content_type = "application/json");
            session_result post(const string& path, 
                                const MultipartFormDataItems& items);
            session_result del(const string& path);                
    };

    inline Session::Session(const string& scheme_host_port, bool verbose /* = false */) : 
        cli_{scheme_host_port} {
        if (verbose) {
            cli_.set_logger([](const Request& req, const Response& resp) {
                cout << endl;
                cout << req.method << " " << req.path << endl;
                for (auto header: req.headers) {
                    cout << header.first << ": " << header.second << endl;
                }
                cout << endl;
                std::cout << req.body << std::endl;
                cout << endl << endl;

                cout << resp.status << " " << resp.reason << endl;
                for (auto header: resp.headers) {
                    cout << header.first << ": " << header.second << endl;
                }
                cout << endl;
                std::cout << resp.body << std::endl;
                cout << endl << endl;
            });
        }
    }

    inline void Session::stop() {
        cli_.stop();
    }
    
    inline void Session::set_token(const string& token) {
        cli_.set_bearer_token_auth(token);
    }

    inline void Session::set_proxy(const string& host, int port) {
        cli_.set_proxy(host, port);
    }

    inline session_result Session::get(const string& path) {
        auto res = cli_.Get(path);
        if (res.error() != Error::Success) {
            return make_tuple(-1, httplib::to_string(res.error()));
        }
        if (res->status != StatusCode::OK_200) {
            return make_tuple(-1, res->reason);
        }
        return make_tuple(0, res->body);
    }

    inline session_result Session::post(const string& path, 
                                 const string& data, 
                                 const string& content_type /* = "application/json" */) {
        auto res = cli_.Post(path, data, content_type);
        if (res.error() != Error::Success) {
            return make_tuple(-1, httplib::to_string(res.error()));
        }
        if (res->status != StatusCode::OK_200) {
            return make_tuple(-1, res->reason);
        }
        return make_tuple(0, res->body);
    }

    inline session_result Session::post(const string& path, 
                                 const MultipartFormDataItems& items) {
        auto res = cli_.Post(path, items);
        if (res.error() != Error::Success) {
            return make_tuple(-1, httplib::to_string(res.error()));
        }
        if (res->status != StatusCode::OK_200) {
            return make_tuple(-1, res->reason);
        }
        return make_tuple(0, res->body);
    }

    inline session_result Session::del(const string& path) {
        auto res = cli_.Delete(path);
        if (res.error() != Error::Success) {
            return make_tuple(-1, httplib::to_string(res.error()));
        }
        if (res->status != StatusCode::OK_200) {
            return make_tuple(-1, res->reason);
        }
        return make_tuple(0, res->body);
    }

    class OpenAI;

    inline string file_content(const string& path) {
        ifstream is(path, ios::binary);
        ostringstream oss;
        oss << is.rdbuf();
        return oss.str();
    }

    class CategoryAudio {
        OpenAI& openai_;

        public:
            CategoryAudio(OpenAI& openai) : 
                openai_{openai} {}
            
            string speech(json request);
            json transcription(json request);
            json translation(json request);
    };

    class CategoryChat {
        OpenAI& openai_;

        public:
            CategoryChat(OpenAI& openai) : 
                openai_{openai} {}

            json create(json request);
    };

    class CategoryEmbedding {
        OpenAI& openai_;

        public:
            CategoryEmbedding(OpenAI& openai) : 
                openai_{openai} {}
            
            json create(json request);
    };

    class CategoryFinetunning {
        OpenAI& openai_;

        public:
            CategoryFinetunning(OpenAI& openai) : 
                openai_{openai} {}

            json create(json request);
            json list();
            json events(const string& fine_tuning_job_id);
            json checkpoints(const string& fine_tuning_job_id);
            json retrieve(const string& fine_tuning_job_id);
            json cancel(const string& fine_tuning_job_id);
    };

    class CategoryFiles {
        OpenAI& openai_;

        public:
            CategoryFiles(OpenAI& openai) : 
                openai_{openai} {}

            json upload(json request);
            json list();
            json retrieve(const string& file_id);
            json del(const string& file_id);
            json content(const string& file_id);
    };

    class CategoryImages {
        OpenAI& openai_;

        public:
            CategoryImages(OpenAI& openai) : 
                openai_{openai} {}

            json create(json request);
            json edit(json request);
            json variation(json request);
    };

    class CategoryModels {
        OpenAI& openai_;

        public:
            CategoryModels(OpenAI& openai) : 
                openai_{openai} {}

            json list();
            json retrieve(const string& model);
            json del(const string& model);
    };

    class CategoryModerations {
        OpenAI& openai_;

        public:
            CategoryModerations(OpenAI& openai) : 
                openai_{openai} {}

            json create(json request);
    };

    class OpenAI {
        Session session_;

        public:
            OpenAI(const string& scheme_host_port, 
                   const string& token = "", 
                   const string& proxy_host_port = "", 
                   bool verbose = false);

            void stop();

            json get(const string& path);
            json post(const string& path, 
                      const string& data, 
                      const string& content_type = "application/json");
            json post(const string& path, 
                      const MultipartFormDataItems& items);
            json del(const string& path);

        public:
            CategoryAudio audio { *this };
            CategoryChat chat { *this };
            CategoryEmbedding embedding { *this };
            CategoryFiles files { *this };
            CategoryFinetunning finetunning { *this };
            CategoryImages images { *this };
            CategoryModerations moderations { *this };
            CategoryModels models { *this };
    };

    inline OpenAI::OpenAI(const string& scheme_host_port, 
                   const string& token /* = "" */, 
                   const string& proxy_host_port /* = "" */, 
                   bool verbose /* = false */)
            : session_{scheme_host_port, verbose} {
        if (!token.empty()) {
            session_.set_token(token);
        }

        if (!proxy_host_port.empty()) {
            int n = proxy_host_port.find(":");
            if (n != string::npos) {
                session_.set_proxy(proxy_host_port.substr(0, n), 
                                   stoi(proxy_host_port.substr(n + 1)));
            }
        }
    }

    inline void OpenAI::stop() {
        session_.stop();
    }

    inline json OpenAI::get(const string& path) {
        int result;
        string response;

        tie(result, response) = session_.get(path);
        if (result) {
            throw runtime_error(response);
        }

        try {
            return json::parse(response);
        } catch (const exception& ) {
            return {{ "response", response }};
        }
    }

    inline json OpenAI::post(const string& path, 
                      const string& data, 
                      const string& content_type /* = "application/json" */) {
        int result;
        string response;

        tie(result, response) = session_.post(path, data, content_type);
        if (result) {
            throw runtime_error(response);
        }

        try {
            return json::parse(response);
        } catch (const exception& ) {
            return {{ "response", response }};
        }
    }

    inline json OpenAI::post(const string& path, 
                      const MultipartFormDataItems& items) {
        int result;
        string response;

        tie(result, response) = session_.post(path, items);
        if (result) {
            throw runtime_error(response);
        }

        try {
            return json::parse(response);
        } catch (const exception& ) {
            return {{ "response", response }};
        }
    }

    inline json OpenAI::del(const string& path) {
        int result;
        string response;

        tie(result, response) = session_.del(path);
        if (result) {
            throw runtime_error(response);
        }

        try {
            return json::parse(response);
        } catch (const exception& ) {
            return {{ "response", response }};
        }
    }

    inline string CategoryAudio::speech(json request) {
        json res = openai_.post("/v1/audio/speech", request.dump());
        return res["response"].get<string>();
    }

    inline json CategoryAudio::transcription(json request) {
        MultipartFormDataItems items;

        if (request.contains("file")) {
            string path = request["file"].get<string>();
            string content = file_content(path);
            items.push_back({"file", content, path, "audio/mpeg"});
        }

        if (request.contains("model")) {
            string model = request["model"].get<string>();
            items.push_back({"model", model, "", ""});
        }

        if (request.contains("language")) {
            string language = request["language"].get<string>();
            items.push_back({"language", language, "", ""});
        }

        if (request.contains("prompt")) {
            string prompt = request["prompt"].get<string>();
            items.push_back({"prompt", prompt, "", ""});
        }

        if (request.contains("response_format")) {
            string response_format = request["response_format"].get<string>();
            items.push_back({"response_format", response_format, "", ""});
        }

        if (request.contains("temperature")) {
            string temperature = to_string(request["temperature"].get<float>());
            items.push_back({"temperature", temperature, "", ""});
        }

        return openai_.post("/v1/audio/transcriptions", items);
    }

    inline json CategoryAudio::translation(json request) {
        MultipartFormDataItems items;

        if (request.contains("file")) {
            string path = request["file"].get<string>();
            string content = file_content(path);
            items.push_back({"file", content, path, "audio/mpeg"});
        }

        if (request.contains("model")) {
            string model = request["model"].get<string>();
            items.push_back({"model", model, "", ""});
        }

        if (request.contains("prompt")) {
            string prompt = request["prompt"].get<string>();
            items.push_back({"prompt", prompt, "", ""});
        }

        if (request.contains("response_format")) {
            string response_format = request["response_format"].get<string>();
            items.push_back({"response_format", response_format, "", ""});
        }

        if (request.contains("temperature")) {
            string temperature = to_string(request["temperature"].get<float>());
            items.push_back({"temperature", temperature, "", ""});
        }

        return openai_.post("/v1/audio/translations", items);
    }

    inline json CategoryChat::create(json request) {
        return openai_.post("/v1/chat/completions", request.dump());
    }

    inline json CategoryEmbedding::create(json request) {
        return openai_.post("/v1/embeddings", request.dump());
    }

    inline json CategoryFinetunning::create(json request) {
        return openai_.post("/v1/fine_tuning/jobs", request.dump());
    }

    inline json CategoryFinetunning::list() {
        return openai_.get("/v1/fine_tuning/jobs");
    }

    inline json CategoryFinetunning::events(const string& fine_tuning_job_id) {
        return openai_.get(string("/v1/fine_tuning/jobs/") + fine_tuning_job_id + "/events");
    }

    inline json CategoryFinetunning::checkpoints(const string& fine_tuning_job_id) {
        return openai_.get(string("/v1/fine_tuning/jobs/") + fine_tuning_job_id + "/checkpoints");
    }

    inline json CategoryFinetunning::retrieve(const string& fine_tuning_job_id) {
        return openai_.get(string("/v1/fine_tuning/jobs/") + fine_tuning_job_id);
    }

    inline json CategoryFinetunning::cancel(const string& fine_tuning_job_id) {
        return openai_.post(string("/v1/fine_tuning/jobs/") + fine_tuning_job_id + "/cancel", "");
    }

    inline json CategoryFiles::upload(json request) {
        MultipartFormDataItems items;

        if (request.contains("file")) {
            string path = request["file"].get<string>();
            string content = file_content(path);
            items.push_back({"file", content, path, "application/json"});
        }

        if (request.contains("purpose")) {
            string purpose = request["purpose"].get<string>();
            items.push_back({"purpose", purpose, "", ""});
        }

        return openai_.post("/v1/files", items);
    }

    inline json CategoryFiles::list() {
        return openai_.get("/v1/files");
    }

    inline json CategoryFiles::retrieve(const string& file_id) {
        return openai_.get(string("/v1/files/") + file_id);
    }

    inline json CategoryFiles::del(const string& file_id) {
        return openai_.del(string("/v1/files/") + file_id);
    }

    inline json CategoryFiles::content(const string& file_id) {
        return openai_.get(string("/v1/files/") + file_id + "/content");
    }

    inline json CategoryImages::create(json request) {
        return openai_.post("/v1/images/generations", request.dump());
    }

    inline json CategoryImages::edit(json request) {
        MultipartFormDataItems items;

        if (request.contains("image")) {
            string path = request["image"].get<string>();
            string content = file_content(path);
            items.push_back({"image", content, path, "image/png"});
        }

        if (request.contains("prompt")) {
            string prompt = request["prompt"].get<string>();
            items.push_back({"prompt", prompt, "", ""});
        }

        if (request.contains("mask")) {
            string path = request["mask"].get<string>();
            string content = file_content(path);
            items.push_back({"mask", content, path, "image/png"});
        }

        if (request.contains("model")) {
            string model = request["model"].get<string>();
            items.push_back({"model", model, "", ""});
        }

        if (request.contains("n")) {
            string n = to_string(request["n"].get<int>());
            items.push_back({"n", n, "", ""});
        }

        if (request.contains("size")) {
            string size = to_string(request["size"].get<int>());
            items.push_back({"size", size, "", ""});
        }

        if (request.contains("response_format")) {
            string response_format = request["response_format"].get<string>();
            items.push_back({"response_format", response_format, "", ""});
        }

        if (request.contains("user")) {
            string user = request["user"].get<string>();
            items.push_back({"user", user, "", ""});
        }

        return openai_.post("/v1/images/edits", items);
    }

    inline json CategoryImages::variation(json request) {
        MultipartFormDataItems items;

        if (request.contains("image")) {
            string path = request["image"].get<string>();
            string content = file_content(path);
            items.push_back({"image", content, path, "image/png"});
        }

        if (request.contains("model")) {
            string model = request["model"].get<string>();
            items.push_back({"model", model, "", ""});
        }

        if (request.contains("n")) {
            string n = to_string(request["n"].get<int>());
            items.push_back({"n", n, "", ""});
        }

        if (request.contains("response_format")) {
            string response_format = request["response_format"].get<string>();
            items.push_back({"response_format", response_format, "", ""});
        }

        if (request.contains("size")) {
            string size = to_string(request["size"].get<int>());
            items.push_back({"size", size, "", ""});
        }

        if (request.contains("user")) {
            string user = request["user"].get<string>();
            items.push_back({"user", user, "", ""});
        }

        return openai_.post("/v1/images/variations", items);
    }

    inline json CategoryModels::list() {
        return openai_.get("/v1/models");
    }

    inline json CategoryModels::retrieve(const string& model) {
        return openai_.get(string("/v1/models/") + model);
    }

    inline json CategoryModels::del(const string& model) {
        return openai_.del(string("/v1/models/") + model);
    }

    inline json CategoryModerations::create(json request) {
        return openai_.post("/v1/moderations", request.dump());
    }

    inline OpenAI& start(const string& scheme_host_port = "", 
                  const string& token = "", 
                  const string& proxy_host_port = "",
                  const bool verbose = false) {
        static OpenAI instance(scheme_host_port, token, proxy_host_port, verbose);
        return instance;
    }

    inline OpenAI& instance() {
        return start();
    }

    inline void stop() {
        instance().stop();
    }

    inline CategoryAudio& audio() {
        return instance().audio;
    }

    inline CategoryChat& chat() {
        return instance().chat;
    }

    inline CategoryEmbedding& embedding() {
        return instance().embedding;
    }

    inline CategoryFiles& files() {
        return instance().files;
    }

    inline CategoryFinetunning& finetunning() {
        return instance().finetunning;
    }

    inline CategoryImages& images() {
        return instance().images;
    }

    inline CategoryModerations& moderations() {
        return instance().moderations;
    }

    inline CategoryModels& models() {
        return instance().models;
    }
}