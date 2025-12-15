//===-- Lexer.h ------------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_LEXER_H
#define EZ_COMPILE_LEXER_H

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace ezcompile {

/// Structure definition a location in a file.
struct Location {
	std::shared_ptr<std::string> file; ///< filename.
	int line;                          ///< line number.
	int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
	tok_semicolon        = ';',
	tok_comma			 = ',',
	tok_parenthese_open  = '(',
	tok_parenthese_close = ')',
	tok_bracket_open     = '{',
	tok_bracket_close    = '}',
	tok_sbracket_open    = '[',
	tok_sbracket_close   = ']',

	// Operators (returned as their ASCII value, but listed here for clarity if needed)
	tok_assign = '=',
	tok_plus   = '+',
	tok_minus  = '-',
	tok_mul    = '*',
	tok_div    = '/',

	tok_eof = -1,

	// keywords for comp language
	tok_declarations = -2,
	tok_equations    = -3,
	tok_options      = -4,
	tok_diff		 = -5,

	// primary
	tok_identifier = -6,
	tok_number     = 7,
};

class Lexer {
public:
	explicit Lexer(std::string filename)
		: lastLocation({std::make_shared<std::string>(std::move(filename)), 0, 0}) {

	}

	virtual ~Lexer() = default;

	/// Look at the current token in the stream.
	Token getCurToken() { return curTok; }

	/// Move to the next token in the stream and return it.
	Token getNextToken() { return curTok = getTok(); }

	/// Move to the next token in the stream, asserting on the current token
	/// matching the expectation.
	void consume(Token tok) {
		assert(tok == curTok && "consume Token mismatch expectation");
		getNextToken();
	}

	/// Return the current identifier (prereq: getCurToken() == tok_identifier)
	llvm::StringRef getId() {
		assert(curTok == tok_identifier);
		return identifierStr;
	}

	/// Return the current number (prereq: getCurToken() == tok_number)
	double getValue() {
		assert(curTok == tok_number);
		return numVal;
	}

	/// Return the location for the beginning of the current token.
	Location getLastLocation() { return lastLocation; }

	// Return the current line in the file.
	[[nodiscard]] int getLine() const { return curLineNum; }

	// Return the current column in the file.
	[[nodiscard]] int getCol() const { return curCol; }

private:
	/// Delegate to a derived class fetching the next line.
	virtual llvm::StringRef readNextLine() = 0;

	/// Return the next character from the stream.
	int getNextChar();

	///  Return the next token from standard input.
	Token getTok();

	/// The last token read from the input.
	Token curTok = tok_eof;

	/// Location for `curTok`.
	Location lastLocation;

	/// If the current Token is an identifier, this string contains the value.
	std::string identifierStr;

	/// If the current Token is a number, this contains the value.
	double numVal = 0;

	/// The last value returned by getNextChar().
	Token lastChar = Token(' ');

	/// Keep track of the current line number in the input stream
	int curLineNum = 0;

	/// Keep track of the current column number in the input stream
	int curCol = 0;

	/// Buffer supplied by the derived class on calls to `readNextLine()`
	llvm::StringRef curLineBuffer = "\n";
};

}

#endif //EZ_COMPILE_LEXER_H
