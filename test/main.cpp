#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

#include "../include/openai.h"

using namespace std;

namespace po = boost::program_options;

void usage(const string& name, const po::options_description& opts) {
    cout << "usage: " << endl 
         << name << " [--help|audio|chat|embedding|fine-tunning|files|images|models|moderations]" 
         << " [--speech|transcription|translation|create|list|events|checkpoints|retrieve|cancel|upload|delete|edit|variation]" 
         << " [--data data]"
         << " [--base-uri schema://host:port]"
         << " [--token token]"
         << " [--proxy host:port]"
         << endl << endl
         <<"options: " << endl
         << opts << endl;
}

int main(int argc, char * argv[]) {
    po::options_description opts;
    opts.add_options()
                    ("help,h", "show this help message and exit")
                    ("base-uri", po::value<string>()->default_value("https://api.openai.com"), "schema://host:port")
                    ("token", po::value<string>()->default_value(""), "token")
                    ("proxy", po::value<string>()->default_value(""), "host:port")
                    ("audio", "turn audio into text or text into audio.")
                    ("chat,c", "given a list of messages comprising a conversation, the model will return a response.")
                    ("embedding", "get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.")
                    ("fine-tunning", "manage fine-tuning jobs to tailor a model to your specific training data.")
                    ("files", "files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.")
                    ("images", "given a prompt and/or an input image, the model will generate a new image.")
                    ("models,m", "list and describe the various models available in the API.")
                    ("moderations", "given some input text, outputs if the model classifies it as potentially harmful across several categories.")
                    ("speech", "[--audio] generates audio from the input text.")
                    ("transcription", "[--audio] transcribes audio into the input language.")
                    ("translation", "[--audio] translates audio into english.")
                    ("create", "[--chat] creates a model response for the given chat conversation.\n"
                               "[--embedding] creates an embedding vector representing the input text.\n"
                               "[--fine-tunning] creates a fine-tuning job which begins the process of creating a new model from a given dataset.\n"
                               "[--images] creates an image given a prompt.\n"
                               "[--moderations] given some input text, outputs if the model classifies it as potentially harmful across several categories.\n"
                               )
                    ("list", "[--fine-tunning] list your organization's fine-tuning jobs\n"
                             "[--files] returns a list of files that belong to the user's organization.\n"
                             "[--models] lists the currently available models, and provides basic information about each one such as the owner and availability.\n"
                             )
                    ("events", "[--fine-tunning] get status updates for a fine-tuning job.")
                    ("checkpoints", "[--fine-tunning] list checkpoints for a fine-tuning job.")
                    ("retrieve", "[--fine-tunning] get info about a fine-tuning job.\n"
                                 "[--files] returns information about a specific file.\n"
                                 "[--models] retrieves a model instance, providing basic information about the model such as the owner and permissioning.\n"
                                 )
                    ("cancel", "[--fine-tunning] immediately cancel a fine-tune job.")
                    ("upload", "[--files] upload a file that can be used across various endpoints. Individual files can be up to 512 MB, and the size of all files uploaded by one organization can be up to 100 GB.")
                    ("delete", "[--files] delete a file.\n"
                               "[--models] delete a fine-tuned model. You must have the Owner role in your organization to delete a model.\n"
                               )
                    ("edit", "[--images] creates an edited or extended image given an original image and a prompt.")
                    ("variation", "[--images] creates a variation of a given image.")
                    ("data,d", po::value<string>(), "body of the request.")
                    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);

    if (vm.count("help") > 0) {
        usage(argv[0], opts);
        return 0;
    }

    try {
        openai::start(vm["base-uri"].as<string>(), 
                      vm["token"].as<string>(), 
                      vm["proxy"].as<string>());
        if (vm.count("audio") > 0) {
            if (vm.count("speech") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        string response = openai::audio().speech(json::parse(data));
                        ofstream o("output/result.mp3", ios::binary);
                        o << response;
                        o.close();

                        cout << "output/result.mp3 ok!" << endl;
                    }                    
                }
            } else if (vm.count("transcription") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::audio().transcription(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            } else if (vm.count("translation") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::audio().translation(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            }
        } else if (vm.count("chat") > 0) {
            if (vm.count("create") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;
                        json response = openai::chat().create(json::parse(data.str()));
                        cout << response.dump() << endl;
                    }
                }
            }
        } else if (vm.count("embedding") > 0) {
            if (vm.count("create") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::embedding().create(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            }
        } else if (vm.count("fine-tunning") > 0) {
            if (vm.count("create") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::finetunning().create(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            } else if (vm.count("list") > 0) {
                json response = openai::finetunning().list();
                cout << response.dump() << endl;
            } else if (vm.count("events") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::finetunning().events(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            } else if (vm.count("checkpoints") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::finetunning().checkpoints(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            } else if (vm.count("retrieve") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::finetunning().retrieve(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            } else if (vm.count("cancel") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::finetunning().cancel(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            }
        } else if (vm.count("files") > 0) {
            if (vm.count("upload") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::files().upload(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            } else if (vm.count("list") > 0) {
                json response = openai::files().list();
                cout << response.dump() << endl;
            } else if (vm.count("retrieve") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::files().retrieve(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            } else if (vm.count("delete") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::files().del(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            }
        } else if (vm.count("images") > 0) {
            if (vm.count("create") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::images().create(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            } else if (vm.count("edit") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::images().edit(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            } else if (vm.count("variation") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::images().variation(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            }
        } else if (vm.count("models") > 0) {
            if (vm.count("list") > 0) {
                json response = openai::models().list();
                cout << response.dump() << endl;
            } else if (vm.count("retrieve") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::models().retrieve(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            } else if (vm.count("delete") > 0) {
                if (vm.count("data") > 0) {
                    json response = openai::models().del(vm["data"].as<string>());
                    cout << response.dump() << endl;
                }
            }
        } else if (vm.count("moderations") > 0) {
            if (vm.count("create") > 0) {
                if (vm.count("data") > 0) {
                    ifstream is(vm["data"].as<string>());
                    if (is.is_open()) {
                        stringstream data;
                        data << is.rdbuf();
                        cout << "data: "  << endl << data.str() << endl;

                        json response = openai::moderations().create(json::parse(data));
                        cout << response.dump() << endl;
                    }
                }
            }
        } else {
            usage(argv[0], opts);
        }
    } catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
    }
    
    return 0;
}