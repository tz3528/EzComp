//===-- Lexer.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"

namespace ezcompile {

int Lexer::getNextChar() {
	if (curLineBuffer.empty())
		return EOF;
	++curCol;
	auto nextchar = curLineBuffer.front();
	curLineBuffer = curLineBuffer.drop_front();
	if (curLineBuffer.empty())
		curLineBuffer = readNextLine();
	if (nextchar == '\n') {
		++curLineNum;
		curCol = 0;
	}
	return nextchar;
}

Token Lexer::getTok() {
	// Skip any whitespace.
	while (isspace(lastChar))
		lastChar = Token(getNextChar());

	// Save the current location.
	lastLocation.line = curLineNum;
	lastLocation.col = curCol;

	// Identifier: [a-zA-Z][a-zA-Z0-9_]*
	if (isalpha(lastChar)) {
		identifierStr = (char)lastChar;
		while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
			identifierStr += (char)lastChar;

		// Check for comp language keywords
		if (identifierStr == "declarations")
			return tok_declarations;
		if (identifierStr == "equations")
			return tok_equations;
		if (identifierStr == "options")
			return tok_options;
		if (identifierStr == "diff")
			return tok_diff;

		return tok_identifier;
	}

	// Number: [0-9.]+
	// Note: This parses positive numbers.
	// Negative numbers (e.g. -6) are parsed as tok_minus then tok_number.
	if (isdigit(lastChar) || lastChar == '.') {
		std::string numStr;
		do {
			numStr += std::to_string(lastChar);
			lastChar = Token(getNextChar());
		}
		while (isdigit(lastChar) || lastChar == '.');

		numVal = strtod(numStr.c_str(), nullptr);
		return tok_number;
	}

	// Comment support: '#' until end of line
	if (lastChar == '#') {
		do {
			lastChar = Token(getNextChar());
		}
		while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

		if (lastChar != EOF)
			return getTok();
	}

	// Check for end of file.
	if (lastChar == EOF)
		return tok_eof;

	// Otherwise, return the character as its ascii value.
	// This handles +, -, =, ;, etc.
	auto thisChar = Token(lastChar);
	lastChar = Token(getNextChar());
	return thisChar;
}


}