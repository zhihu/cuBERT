#include "gtest/gtest.h"

#include "cuBERT/tokenization.h"

class TokenizationTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TokenizationTest, full_tokenizer) {
    cuBERT::FullTokenizer tokenizer("test_vocab.txt");

    std::vector<std::string> tokens;
    tokenizer.tokenize(u8"UNwant\u00E9d,running", &tokens, 32);

    EXPECT_STREQ(tokens[0].c_str(), "un");
    EXPECT_STREQ(tokens[1].c_str(), "##want");
    EXPECT_STREQ(tokens[2].c_str(), "##ed");
    EXPECT_STREQ(tokens[3].c_str(), ",");
    EXPECT_STREQ(tokens[4].c_str(), "runn");
    EXPECT_STREQ(tokens[5].c_str(), "##ing");

    uint64_t ids[6];
    tokenizer.convert_tokens_to_ids(tokens, ids);

    EXPECT_EQ(ids[0], 7);
    EXPECT_EQ(ids[1], 4);
    EXPECT_EQ(ids[2], 5);
    EXPECT_EQ(ids[3], 10);
    EXPECT_EQ(ids[4], 8);
    EXPECT_EQ(ids[5], 9);
}

TEST_F(TokenizationTest, chinese) {
    cuBERT::BasicTokenizer tokenizer;

    std::vector<std::string> tokens;
    tokenizer.tokenize(u8"ah\u535A\u63A8zz", &tokens, 32);

    EXPECT_STREQ(tokens[0].c_str(), u8"ah");
    EXPECT_STREQ(tokens[1].c_str(), u8"\u535A");
    EXPECT_STREQ(tokens[2].c_str(), u8"\u63A8");
    EXPECT_STREQ(tokens[3].c_str(), u8"zz");
}

TEST_F(TokenizationTest, basic_tokenizer_lower) {
    cuBERT::BasicTokenizer tokenizer(true);

    std::vector<std::string> tokens;
    tokenizer.tokenize(u8" \tHeLLo!how  \n Are yoU?  ", &tokens, 32);

    EXPECT_STREQ(tokens[0].c_str(), "hello");
    EXPECT_STREQ(tokens[1].c_str(), "!");
    EXPECT_STREQ(tokens[2].c_str(), "how");
    EXPECT_STREQ(tokens[3].c_str(), "are");
    EXPECT_STREQ(tokens[4].c_str(), "you");
    EXPECT_STREQ(tokens[5].c_str(), "?");

    tokens.clear();
    tokenizer.tokenize(u8"H\u00E9llo", &tokens, 32);
    EXPECT_STREQ(tokens[0].c_str(), "hello");
}

TEST_F(TokenizationTest, basic_tokenizer_no_lower) {
    cuBERT::BasicTokenizer tokenizer(false);

    std::vector<std::string> tokens;
    tokenizer.tokenize(u8" \tHeLLo!how  \n Are yoU?  ", &tokens, 32);

    EXPECT_STREQ(tokens[0].c_str(), "HeLLo");
    EXPECT_STREQ(tokens[1].c_str(), "!");
    EXPECT_STREQ(tokens[2].c_str(), "how");
    EXPECT_STREQ(tokens[3].c_str(), "Are");
    EXPECT_STREQ(tokens[4].c_str(), "yoU");
    EXPECT_STREQ(tokens[5].c_str(), "?");
}

TEST_F(TokenizationTest, wordpiece_tokenizer) {
    std::unordered_map<std::string, uint64_t> vocab;
    cuBERT::load_vocab("test_vocab.txt", &vocab);

    cuBERT::WordpieceTokenizer tokenizer(&vocab);

    std::vector<std::string> tokens;
    tokenizer.tokenize("", &tokens);

    EXPECT_EQ(tokens.size(), 0);

    tokens.clear();
    tokenizer.tokenize("unwanted", &tokens);
    tokenizer.tokenize("running", &tokens);

    EXPECT_STREQ(tokens[0].c_str(), "un");
    EXPECT_STREQ(tokens[1].c_str(), "##want");
    EXPECT_STREQ(tokens[2].c_str(), "##ed");
    EXPECT_STREQ(tokens[3].c_str(), "runn");
    EXPECT_STREQ(tokens[4].c_str(), "##ing");

    tokens.clear();
    tokenizer.tokenize("unwantedX", &tokens);
    tokenizer.tokenize("running", &tokens);

    EXPECT_STREQ(tokens[0].c_str(), "[UNK]");
    EXPECT_STREQ(tokens[1].c_str(), "runn");
    EXPECT_STREQ(tokens[2].c_str(), "##ing");
}

TEST_F(TokenizationTest, convert_tokens_to_ids) {
    cuBERT::FullTokenizer tokenizer("test_vocab.txt");

    std::vector<std::string> tokens = {"un", "##want", "##ed", "runn", "##ing"};
    uint64_t ids[5];
    tokenizer.convert_tokens_to_ids(tokens, ids);

    EXPECT_EQ(ids[0], 7);
    EXPECT_EQ(ids[1], 4);
    EXPECT_EQ(ids[2], 5);
    EXPECT_EQ(ids[3], 8);
    EXPECT_EQ(ids[4], 9);
}

TEST_F(TokenizationTest, is_whitespace) {
    EXPECT_TRUE(cuBERT::_is_whitespace(' '));
    EXPECT_TRUE(cuBERT::_is_whitespace('\t'));
    EXPECT_TRUE(cuBERT::_is_whitespace('\r'));
    EXPECT_TRUE(cuBERT::_is_whitespace('\n'));
    EXPECT_TRUE(cuBERT::_is_whitespace(0x00A0));

    EXPECT_FALSE(cuBERT::_is_whitespace('A'));
    EXPECT_FALSE(cuBERT::_is_whitespace('-'));
}

TEST_F(TokenizationTest, is_control) {
    EXPECT_TRUE(cuBERT::_is_control(0x0005));

    EXPECT_FALSE(cuBERT::_is_control('A'));
    EXPECT_FALSE(cuBERT::_is_control(' '));
    EXPECT_FALSE(cuBERT::_is_control('\t'));
    EXPECT_FALSE(cuBERT::_is_control('\r'));
}

TEST_F(TokenizationTest, is_punctuation) {
    EXPECT_TRUE(cuBERT::_is_punctuation('-'));
    EXPECT_TRUE(cuBERT::_is_punctuation('$'));
    EXPECT_TRUE(cuBERT::_is_punctuation('`'));
    EXPECT_TRUE(cuBERT::_is_punctuation('.'));

    EXPECT_FALSE(cuBERT::_is_punctuation('A'));
    EXPECT_FALSE(cuBERT::_is_punctuation(' '));
}
