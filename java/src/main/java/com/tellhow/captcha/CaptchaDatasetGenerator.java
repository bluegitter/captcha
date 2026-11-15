package com.tellhow.captcha;

import com.google.code.kaptcha.Producer;
import com.google.code.kaptcha.impl.DefaultKaptcha;
import com.google.code.kaptcha.util.Config;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.util.Properties;

/** Generates labelled images with the production Kaptcha settings copied locally. */
public final class CaptchaDatasetGenerator {
    private CaptchaDatasetGenerator() { }

    /**
     * Keep these values aligned with ClientCacheConfig/KaptchaProperties in the
     * production service. The generator deliberately has no cloud-common-captcha
     * dependency so it can be built as a standalone project.
     */
    private static Producer createProducer() throws Exception {
        Properties settings = new Properties();
        try (InputStream input = CaptchaDatasetGenerator.class.getClassLoader()
                .getResourceAsStream("captcha.properties")) {
            if (input == null) {
                throw new IllegalStateException("captcha.properties is missing from the classpath");
            }
            settings.load(input);
        }
        DefaultKaptcha producer = new DefaultKaptcha();
        producer.setConfig(new Config(settings));
        return producer;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 2) {
            throw new IllegalArgumentException("usage: <output-dir> [count]");
        }
        File output = new File(args[0]);
        int count = args.length == 2 ? Integer.parseInt(args[1]) : 3000;
        if (!output.isDirectory() && !output.mkdirs()) {
            throw new IllegalStateException("cannot create output directory: " + output);
        }

        Producer producer = createProducer();
        for (int i = 0; i < count; i++) {
            String text = producer.createText();
            BufferedImage image = producer.createImage(text);
            File target = new File(output, text.toLowerCase() + "_" + String.format("%06d", i) + ".jpg");
            ImageIO.write(image, "jpeg", target);
        }
        System.out.printf("generated %d Java Kaptcha images in %s%n", count, output.getAbsolutePath());
    }
}
