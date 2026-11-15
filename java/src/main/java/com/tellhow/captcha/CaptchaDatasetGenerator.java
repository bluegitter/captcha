package com.tellhow.captcha;

import com.google.code.kaptcha.Producer;
import com.tellhow.cloud.common.captcha.config.KaptchaConfiguration;
import com.tellhow.cloud.common.captcha.config.KaptchaProperties;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.concurrent.ThreadLocalRandom;

/** Generates labelled images with the same Producer used by CodeEndpoint. */
public final class CaptchaDatasetGenerator {
    private CaptchaDatasetGenerator() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 2) {
            throw new IllegalArgumentException("usage: <output-dir> [count]");
        }
        File output = new File(args[0]);
        int count = args.length == 2 ? Integer.parseInt(args[1]) : 3000;
        if (!output.isDirectory() && !output.mkdirs()) {
            throw new IllegalStateException("cannot create output directory: " + output);
        }

        KaptchaProperties properties = new KaptchaProperties();
        Producer producer = new KaptchaConfiguration().producer(properties);
        for (int i = 0; i < count; i++) {
            String text = producer.createText();
            BufferedImage image = producer.createImage(text);
            File target = new File(output, text.toLowerCase() + "_" + String.format("%06d", i) + ".jpg");
            ImageIO.write(image, "jpeg", target);
        }
        System.out.printf("generated %d Java Kaptcha images in %s%n", count, output.getAbsolutePath());
    }
}
