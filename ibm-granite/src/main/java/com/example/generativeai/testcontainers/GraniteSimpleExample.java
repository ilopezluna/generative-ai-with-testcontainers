package com.example.generativeai.testcontainers;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.localai.LocalAiChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.ollama.OllamaContainer;
import org.testcontainers.utility.DockerImageName;

@Slf4j
public class GraniteSimpleExample {

	@SneakyThrows
	public static void main(String[] args) {
		ChatLanguageModel model = getGraniteModel();
		// ChatLanguageModel model = getLLamaModel();
		String response = model.generate("Can you tell me why Testcontainers is awesome in one sentence?");
		log.info("Response from LLM (\uD83E\uDD16)-> {}", response);
	}

	private static ChatLanguageModel getGraniteModel() {
		var granite = new GenericContainer<>("redhat/granite-7b-lab-gguf:latest").withCommand("--serve")
			.withExposedPorts(8080)
			.waitingFor(Wait.forHttp("/models"));

		granite.start();
		return LocalAiChatModel.builder()
			.baseUrl("http://localhost:%d".formatted(granite.getMappedPort(8080)))
			.modelName("/models/granite-7b-lab-Q4_K_M.gguf")
			.temperature(0.0)
			.topP(0.00001)
			.build();
	}

	private static ChatLanguageModel getLLamaModel() {
		var ollamaContainer = new OllamaContainer(
				DockerImageName.parse("ilopezluna/llama3.2:0.3.13-1b").asCompatibleSubstituteFor("ollama/ollama"));
		ollamaContainer.start();
		return OllamaChatModel.builder()
			.baseUrl(ollamaContainer.getEndpoint())
			.modelName("llama3.2:1b")
			.temperature(0.0)
			.topP(0.00001)
			.build();
	}

}
