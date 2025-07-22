#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
  export $(cat .env | sed 's/#.*//g' | xargs)
fi

curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gpt-4o\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"$1\"}
    ]
  }"



#    The command export $(cat .env | sed 's/#.*//g' | xargs) is a common way to load environment variables from
#   a .env file into your current shell session.

#   Here’s how it works, from the inside out:


#    1. `cat .env`
#        * This command reads the content of the .env file and prints it to standard output.


#    2. `| sed 's/#.*//g'`
#        * The | (pipe) sends the output of cat .env to the sed (stream editor) command.
#        * sed 's/#.*//g' finds any line that contains a # and removes the # and everything after it on that
#          line. This is done to strip out comments from the .env file.


#    3. `| xargs`
#        * The output from sed (which is now the .env file content without comments) is piped to xargs.
#        * xargs takes the lines of input and converts them into a single, space-separated line. This is
#          important because the export command expects the variables to be on a single line.


#    4. `export $(...)`
#        * The $(...) is called "command substitution". The shell first executes the entire pipeline inside the
#          parentheses (cat .env | sed ... | xargs).
#        * The output of that pipeline (e.g., OPENAI_API_KEY="YOUR_API_KEY" ANOTHER_VAR="something") is then
#          substituted into the command line.
#        * export is the final command. It takes the substituted string and sets the variables for the current
#          shell and any child processes it starts.


#   In Summary


#   The entire command does the following:
#    1. Reads the .env file.
#    2. Removes any comments.
#    3. Formats the remaining KEY=VALUE pairs into a single line.
#    4. Exports those KEY=VALUE pairs as environment variables in your shell.