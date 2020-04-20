#pragma once

#include <cstring>
#include <map>
#include <vector>
#include <string>

using namespace std;

class Value
{
private:
  string content;

public:

  Value() : content("") {}

  Value(string content_) : content(content_) { }

  double asDouble(double def=0) const
  {
    if (content == "") return def;
    return (double) atof(content.c_str());
  }

  int asInt(int def=0) const
  {
    if (content == "") return def;
    return atoi(content.c_str());
  }

  bool asBool(bool def=false) const
  {
    if (content == "") return def;
    if (content == "0") return false;
    if (content == "false") return false;

    return true;
  }

  string asString(string def="") const
  {
    if (content == "") return def;

    return content;
  }
};

class ArgumentParser
{
private:

  map<string,Value> mapArguments;

  const int iArgC;
  const char** vArgV;
  bool bStrictMode;

public:

  Value operator()(const string arg)
  {
    if (bStrictMode)
    {
      map<string,Value>::const_iterator it = mapArguments.find(arg);

      if (it == mapArguments.end())
      {
        printf("Runtime option NOT SPECIFIED! ABORTING! name: %s\n",arg.data());
        abort();
      }
    }

    return mapArguments[arg];
  }

  ArgumentParser(const int argc, const char ** argv) : mapArguments(), iArgC(argc), vArgV(argv), bStrictMode(false)
  {
    for (int i=1; i<argc; i++)
      if (argv[i][0] == '-')
      {
        string values = "";
        int itemCount = 0;

        for (int j=i+1; j<argc; j++)
          if (argv[j][0] == '-')
            break;
          else
          {
            if (strcmp(values.c_str(), ""))
              values += ' ';

            values += argv[j];
            itemCount++;
          }

        if (itemCount == 0)
          values += '1';
        mapArguments[argv[i]] = Value(values);
        i += itemCount;
      }
  }

  int getargc() const {
    return iArgC;
  }

  const char** getargv() const {
    return vArgV;
  }

  void set_strict_mode()
  {
    bStrictMode = true;
  }

  void unset_strict_mode()
  {
    bStrictMode = false;
  }

  void save_options(string path=".")
  {
    string options;
    for(map<string,Value>::const_iterator it=mapArguments.begin(); it!=mapArguments.end(); it++)
    {
      options+= it->first + " " + it->second.asString() + " ";
    }
    string filepath = (path + "/" + string("argumentparser.log"));
    FILE * f = fopen(filepath.data(), "a");
    if (f == NULL)
    {
      printf("impossible to write %s.\n", filepath.data());
      return;
    }
    fprintf(f, "%s\n", options.data());
    fclose(f);
  }
};
