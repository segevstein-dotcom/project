# Makefile for SOLE C Implementation
CC = gcc
CFLAGS = -Wall -Wextra -O2 -Isrc
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/utils.c
OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/utils.o
TARGET = $(BIN_DIR)/sole_layernorm

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@if not exist "$(BUILD_DIR)" mkdir $(BUILD_DIR)
	@if not exist "$(BIN_DIR)" mkdir $(BIN_DIR)

# Link
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo.
	@echo ========================================
	@echo   Build successful: $(TARGET)
	@echo ========================================

# Compile
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	@echo.
	@echo ========================================
	@echo   Running SOLE LayerNorm...
	@echo ========================================
	@cd . && $(TARGET)

# Clean build artifacts
clean:
	@if exist "$(BUILD_DIR)" rmdir /s /q $(BUILD_DIR)
	@if exist "$(BIN_DIR)" rmdir /s /q $(BIN_DIR)
	@echo Cleaned build artifacts

# Rebuild
rebuild: clean all

.PHONY: all directories run clean rebuild
