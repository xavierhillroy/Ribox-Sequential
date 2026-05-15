# ---- Configuration ----
CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -O2
DEPFLAGS := -MMD -MP

# ---- Files ----
SRCS := LGPEngine.cpp test_bed.cpp Interpreter.cpp Evaluator.cpp Dataset.cpp
OBJS := $(SRCS:.cpp=.o)
DEPS := $(OBJS:.o=.d)
TARGET := lgp_test

# ---- Targets ----
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)

.PHONY: all cleans