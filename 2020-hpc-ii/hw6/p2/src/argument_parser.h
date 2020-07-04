#pragma once

#include <map>
#include <string>

class ArgumentParser
{
private:
    std::map<std::string, std::string> args;

public:
    int operator()(const std::string &key, int def) {
        auto it = args.find(key);
        if (it == args.end())
            return def;
        return std::stoi(it->second);
    }

    double operator()(const std::string &key, double def) {
        auto it = args.find(key);
        if (it == args.end())
            return def;
        return std::stod(it->second);
    }

    ArgumentParser(int argc, const char **argv) {
        if (argc % 2 != 1)
            throw std::invalid_argument("Expected an even number of arguments.");

        for (int i = 1; i < argc; i += 2) {
            if (argv[i][0] != '-')
                throw std::invalid_argument("Flags should start with a -.");
            args[argv[i]] = argv[i + 1];
        }
    }
};
