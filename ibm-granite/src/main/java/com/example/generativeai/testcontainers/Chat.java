package com.example.generativeai.testcontainers;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.localai.LocalAiStreamingChatModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.model.output.Response;
import lombok.extern.slf4j.Slf4j;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.ollama.OllamaContainer;
import org.testcontainers.utility.DockerImageName;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

@Slf4j
public class Chat {

	public static void main(String[] args) {
		StreamingChatLanguageModel model = getGraniteModel();
		// StreamingChatLanguageModel model = getLLamaModel();

		List<ChatMessage> conversation = new ArrayList<>();

		// Set up the scanner to read user input
		Scanner scanner = new Scanner(System.in);
		String userInput;

		System.out.print("\nYou: ");
		// Enter a conversation loop
		while (true) {

			userInput = scanner.nextLine();

			// Exit the loop if the user types 'exit'
			if (userInput.equalsIgnoreCase("exit")) {
				System.out.println("Ending chat session.");
				System.exit(0);
			}

			// Add user message to the conversation
			conversation.add(UserMessage.from(userInput));

			// Generate AI response
			model.generate(conversation, new StreamingResponseHandler<>() {
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
					conversation.add(response.content());
					System.out.println(); // Print newline after AI's response
					System.out.print("\nYou: ");
				}
			});
		}
	}

	private static StreamingChatLanguageModel getGraniteModel() {
		var granite = new GenericContainer<>("redhat/granite-7b-lab-gguf:latest").withCommand("--serve")
			.withExposedPorts(8080)
			.waitingFor(Wait.forHttp("/models"));

		granite.start();
		return LocalAiStreamingChatModel.builder()
			.baseUrl("http://localhost:%d".formatted(granite.getMappedPort(8080)))
			.modelName("/models/granite-7b-lab-Q4_K_M.gguf")
			.temperature(0.0)
			.topP(0.00001)
			.build();
	}

	private static StreamingChatLanguageModel getLLamaModel() {
		OllamaContainer ollamaContainer = new OllamaContainer(
				DockerImageName.parse("ilopezluna/llama3.2:0.3.13-1b").asCompatibleSubstituteFor("ollama/ollama"));
		ollamaContainer.start();

		return OllamaStreamingChatModel.builder()
			.baseUrl(ollamaContainer.getEndpoint())
			.modelName("llama3.2:1b")
			.temperature(0.0)
			.topP(0.00001)
			.build();
	}

}
