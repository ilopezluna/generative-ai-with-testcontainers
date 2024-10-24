package com.example.generativeai.testcontainers;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.model.output.Response;
import lombok.extern.slf4j.Slf4j;
import org.testcontainers.ollama.OllamaContainer;
import org.testcontainers.utility.DockerImageName;

@Slf4j
public class Streaming {

	public static void main(String[] args) {
		OllamaContainer ollamaContainer = new OllamaContainer(
				DockerImageName.parse("ilopezluna/qwen2:0.3.13-0.5b").asCompatibleSubstituteFor("ollama/ollama"));
		ollamaContainer.start();

		StreamingChatLanguageModel model = OllamaStreamingChatModel.builder()
			.baseUrl(ollamaContainer.getEndpoint())
			.modelName("qwen2:0.5b")
			.build();

		model.generate("Give me a detailed and long explanation of why Testcontainers is great",
				new StreamingResponseHandler<>() {
					@Override
					public void onNext(String token) {
						System.out.print(token);
					}

					@Override
					public void onError(Throwable error) {
						System.out.println("Error: " + error.getMessage());
					}

					@Override
					public void onComplete(Response<AiMessage> response) {
						System.exit(0);
					}
				});

	}

}
